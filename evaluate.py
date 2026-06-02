"""
评测模块 — 从结构化字段自动生成测试集，无需人工标注
三个维度: 向量检索 / 知识图谱 / 混合RAG
"""
import re
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from parse_docs import DrugMDParser, DrugDocument
from build_vector_store import VectorStore
from build_knowledge_graph import KnowledgeGraph
from rag_query import HybridRAG


# ═══════════════════════════════════════════
# 1. 测试集自动生成
# ═══════════════════════════════════════════

@dataclass
class TestCase:
    """单条评测用例"""
    case_id: str
    query: str
    expected_drug: str
    expected_section: str = ""          # 期望命中的章节
    category: str = "indication"        # indication / dosage / contraindication / drug_lookup
    difficulty: str = "easy"            # easy / medium / hard


class TestSetBuilder:
    """从结构化字段自动生成评测集，零人工标注"""

    QUERY_TEMPLATES = {
        "indication": [
            "什么药可以治疗{condition}？",
            "治疗{condition}用什么药？",
            "{condition}应该用什么药物？",
            "哪些药品适用于{condition}？",
            "{condition}的治疗药物有哪些？",
        ],
        "dosage": [
            "{drug}的用法用量是什么？",
            "{drug}怎么用？每次多少剂量？",
            "{drug}的给药方式和剂量？",
            "如何使用{drug}？",
        ],
        "contraindication": [
            "哪些患者不能用{drug}？",
            "{drug}有哪些禁忌？",
            "什么情况下禁用{drug}？",
            "{drug}的禁忌症是什么？",
        ],
        "side_effect": [
            "{drug}有什么不良反应？",
            "{drug}的副作用有哪些？",
            "使用{drug}可能出现什么问题？",
        ],
        "drug_lookup": [
            "介绍一下{drug}",
            "{drug}是什么药？",
            "查询{drug}的说明书",
        ],
    }

    def __init__(self, documents: list[DrugDocument], seed: int = 42):
        self.documents = documents
        random.seed(seed)

    def build(self) -> list[TestCase]:
        """生成全部测试用例"""
        cases = []
        for doc in self.documents:
            cases.extend(self._for_drug(doc))
        return cases

    def _for_drug(self, doc: DrugDocument) -> list[TestCase]:
        """为一个药品生成多条测试"""
        cases = []

        # ---- 药品直接查找（简单） ----
        for tmpl in self.QUERY_TEMPLATES["drug_lookup"][:2]:
            cases.append(TestCase(
                case_id=f"{doc.drug_name[:6]}_lookup",
                query=tmpl.format(drug=doc.drug_name[:15]),
                expected_drug=doc.drug_name,
                expected_section="全文",
                category="drug_lookup",
                difficulty="easy",
            ))

        # ---- 适应症 → 药品（中等） ----
        conditions = self._extract_conditions(doc.sections.get("适应症", ""))
        for cond in conditions[:3]:  # 最多3条
            tmpl = random.choice(self.QUERY_TEMPLATES["indication"])
            cases.append(TestCase(
                case_id=f"{doc.drug_name[:6]}_ind_{cond[:6]}",
                query=tmpl.format(condition=cond),
                expected_drug=doc.drug_name,
                expected_section="适应症",
                category="indication",
                difficulty="medium",
            ))

        # ---- 用法用量（简单） ----
        if "用法用量" in doc.sections:
            for tmpl in self.QUERY_TEMPLATES["dosage"][:2]:
                cases.append(TestCase(
                    case_id=f"{doc.drug_name[:6]}_dosage",
                    query=tmpl.format(drug=doc.drug_name[:15]),
                    expected_drug=doc.drug_name,
                    expected_section="用法用量",
                    category="dosage",
                    difficulty="easy",
                ))

        # ---- 禁忌（中等） ----
        if "禁忌" in doc.sections:
            for tmpl in self.QUERY_TEMPLATES["contraindication"][:2]:
                cases.append(TestCase(
                    case_id=f"{doc.drug_name[:6]}_contra",
                    query=tmpl.format(drug=doc.drug_name[:15]),
                    expected_drug=doc.drug_name,
                    expected_section="禁忌",
                    category="contraindication",
                    difficulty="medium",
                ))

        # ---- 不良反应 ----
        if "不良反应" in doc.sections:
            for tmpl in self.QUERY_TEMPLATES["side_effect"][:1]:
                cases.append(TestCase(
                    case_id=f"{doc.drug_name[:6]}_ae",
                    query=tmpl.format(drug=doc.drug_name[:15]),
                    expected_drug=doc.drug_name,
                    expected_section="不良反应",
                    category="side_effect",
                    difficulty="medium",
                ))

        # ---- 较难：只描述症状不提及药品名 ----
        if conditions:
            best_cond = conditions[0]
            # 避免 query 里直接出现药品名
            hard_query = f"患者出现{best_cond}，应该如何处理？"
            if len(hard_query) < 80:
                cases.append(TestCase(
                    case_id=f"{doc.drug_name[:6]}_hard",
                    query=hard_query,
                    expected_drug=doc.drug_name,
                    expected_section="适应症",
                    category="indication",
                    difficulty="hard",
                ))

        return cases

    def _extract_conditions(self, text: str) -> list[str]:
        """从适应症文本提取简短疾病关键词"""
        if not text.strip():
            return []
        conditions = []
        # 按标点切分取短片段
        parts = re.split(r"[，。；;.、\d+\.]", text)
        for p in parts:
            p = p.strip()
            if 3 <= len(p) <= 30 and "适用于" not in p and "用于" not in p:
                conditions.append(p)
        # 去重归并
        seen = set()
        uniq = []
        for c in conditions:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq[:5]


# ═══════════════════════════════════════════
# 2. 向量检索评测
# ═══════════════════════════════════════════

class VectorEvaluator:
    """评测向量检索质量：Recall@K，MRR，NDCG@K"""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.metrics: dict = {}

    def evaluate(self, test_cases: list[TestCase], k_values=(1, 3, 5)) -> dict:
        """对全部用例计算各项指标"""
        results = []
        for tc in test_cases:
            r = self._evaluate_one(tc, max(k_values))
            results.append(r)

        summary = {}
        for k in k_values:
            summary[f"recall@{k}"] = self._mean([r[f"hit@{k}"] for r in results])
            summary[f"mrr@{k}"] = self._mean([r[f"reciprocal_rank@{k}"] for r in results])

        # NDCG (binary relevance: hit=1, miss=0)
        for k in k_values:
            summary[f"ndcg@{k}"] = self._mean([r[f"ndcg@{k}"] for r in results])

        # 按难度拆分
        by_difficulty = defaultdict(list)
        for r in results:
            by_difficulty[r["difficulty"]].append(r["hit@3"])

        for diff, hits in by_difficulty.items():
            summary[f"recall@3_{diff}"] = self._mean(hits)

        # 按类别拆分
        by_category = defaultdict(list)
        for r in results:
            by_category[r["category"]].append(r["hit@3"])
        for cat, hits in by_category.items():
            summary[f"recall@3_{cat}"] = self._mean(hits)

        self.metrics = summary
        return summary

    def _evaluate_one(self, tc: TestCase, max_k: int) -> dict:
        """评测单条"""
        results = self.vs.search(tc.query, n_results=max_k)

        # 检查命中：返回的 drug_name 中是否含目标药品
        hits = []
        ranks = []
        for i, r in enumerate(results):
            hit = self._is_match(r["drug_name"], tc.expected_drug)
            hits.append(1 if hit else 0)
            if hit:
                ranks.append(i + 1)

        # 指标计算
        rr = 1.0 / ranks[0] if ranks else 0.0

        # DCG: 多位命中可累积，但 IDCG 固定为最优单文档@位置1
        dcg = sum(hits[i] / math.log2(i + 2) for i in range(min(len(hits), max_k)))
        idcg = 1.0  # 完美排序：正确答案在第1位
        ndcg = min(dcg / idcg, 1.0) if idcg > 0 else 0

        out = {
            "case_id": tc.case_id,
            "query": tc.query,
            "expected": tc.expected_drug,
            "difficulty": tc.difficulty,
            "category": tc.category,
            "top3_drugs": [r["drug_name"][:20] for r in results[:3]],
            "reciprocal_rank": rr,
            "ndcg@3": ndcg,
        }
        for k in (1, 3, 5):
            out[f"hit@{k}"] = 1 if any(ranks) and min(ranks) <= k else 0
            out[f"reciprocal_rank@{k}"] = rr if rr > 0 else 0
            out[f"ndcg@{k}"] = ndcg

        return out

    def _is_match(self, retrieved: str, expected: str) -> bool:
        """判断检索到的药品是否匹配目标"""
        # 取前6个字符或药品名主体部分
        return retrieved[:10] in expected or expected[:10] in retrieved

    def _mean(self, values: list) -> float:
        return sum(values) / len(values) if values else 0.0


# ═══════════════════════════════════════════
# 3. 知识图谱评测
# ═══════════════════════════════════════════

class KGEvaluator:
    """评测知识图谱质量：实体覆盖率、关系准确率、连通性"""

    def __init__(self, kg: KnowledgeGraph, documents: list[DrugDocument]):
        self.kg = kg
        self.documents = documents
        self.metrics: dict = {}

    def evaluate(self) -> dict:
        """全方位评测图谱"""
        results = {}

        # 3.1 药品覆盖率
        drug_nodes = [n for n, a in self.kg.graph.nodes(data=True) if a.get("type") == "Drug"]
        results["drug_coverage"] = len(drug_nodes)

        # 3.2 逐药品检查：该有的关系是否都抽到了
        per_drug = []
        for doc in self.documents:
            d = self._evaluate_drug(doc)
            if d:
                per_drug.append(d)

        results["drugs_evaluated"] = len(per_drug)

        for key in ["has_indication_relation", "has_side_effect_relation",
                     "has_route_relation", "has_contraindication_relation",
                     "total_relations"]:
            results[key] = self._mean([d.get(key, 0) for d in per_drug])

        # 3.3 图谱密度
        n = self.kg.graph.number_of_nodes()
        e = self.kg.graph.number_of_edges()
        results["graph_density"] = e / (n * (n - 1)) if n > 1 else 0
        results["nodes"] = n
        results["edges"] = e
        results["avg_degree"] = 2 * e / n if n > 0 else 0

        # 3.4 孤立节点比例
        isolated = sum(1 for n in self.kg.graph.nodes() if self.kg.graph.degree(n) == 0)
        results["isolated_ratio"] = isolated / n if n > 0 else 0

        # 3.5 疾病治疗药品种数分布
        disease_treatment_counts = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get("type") == "Disease":
                in_edges = list(self.kg.graph.in_edges(node))
                disease_treatment_counts.append(len(in_edges))
        results["diseases_with_treatment"] = sum(1 for c in disease_treatment_counts if c > 0)
        results["total_diseases"] = len(disease_treatment_counts)
        results["avg_treatments_per_disease"] = (
            self._mean(disease_treatment_counts) if disease_treatment_counts else 0
        )

        self.metrics = results
        return results

    def _evaluate_drug(self, doc: DrugDocument) -> Optional[dict]:
        """评测单个药品的图谱覆盖率"""
        if doc.drug_name not in self.kg.graph:
            return None

        out = {"drug": doc.drug_name[:20]}

        # 药品总关系数
        out["total_relations"] = self.kg.graph.out_degree(doc.drug_name)

        # 适应症关系：有适应症字段则应有 TREATS 边
        has_indication_text = bool(doc.sections.get("适应症", "").strip())
        has_treats_edge = any(
            d.get("relation") == "TREATS"
            for _, _, d in self.kg.graph.out_edges(doc.drug_name, data=True)
        )
        out["has_indication_relation"] = 1 if (not has_indication_text or has_treats_edge) else 0

        # 不良反应关系
        has_ae_text = bool(doc.sections.get("不良反应", "").strip())
        has_ae_edge = any(
            d.get("relation") == "HAS_SIDE_EFFECT"
            for _, _, d in self.kg.graph.out_edges(doc.drug_name, data=True)
        )
        out["has_side_effect_relation"] = 1 if (not has_ae_text or has_ae_edge) else 0

        # 给药途径关系
        has_dosage_text = bool(doc.sections.get("用法用量", "").strip())
        has_route_edge = any(
            d.get("relation") == "ADMINISTERED_VIA"
            for _, _, d in self.kg.graph.out_edges(doc.drug_name, data=True)
        )
        out["has_route_relation"] = 1 if (not has_dosage_text or has_route_edge) else 0

        # 禁忌关系
        has_contra_text = bool(doc.sections.get("禁忌", "").strip())
        has_contra_edge = any(
            d.get("relation") == "CONTRAINDICATED_FOR"
            for _, _, d in self.kg.graph.out_edges(doc.drug_name, data=True)
        )
        out["has_contraindication_relation"] = 1 if (not has_contra_text or has_contra_edge) else 0

        return out

    def _mean(self, values: list) -> float:
        return sum(values) / len(values) if values else 0.0


# ═══════════════════════════════════════════
# 4. 混合 RAG 端到端评测
# ═══════════════════════════════════════════

class RAGEvaluator:
    """评测混合RAG系统端到端质量"""

    def __init__(self, rag: HybridRAG):
        self.rag = rag
        self.metrics: dict = {}

    def evaluate(self, test_cases: list[TestCase]) -> dict:
        """端到端评测"""
        results = []
        for tc in test_cases:
            r = self._evaluate_one(tc)
            results.append(r)

        # 汇总
        summary = {}
        summary["hit_rate"] = self._mean([r["hit"] for r in results])
        summary["section_match_rate"] = self._mean([r["section_match"] for r in results])
        summary["avg_vector_score"] = self._mean([r["top_vector_score"] for r in results])
        summary["avg_context_length"] = self._mean([r["context_length"] for r in results])
        summary["total_cases"] = len(results)

        # 按难度
        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r["hit"])
        for diff, hits in by_diff.items():
            summary[f"hit_rate_{diff}"] = self._mean(hits)

        self.metrics = summary
        return summary

    def _evaluate_one(self, tc: TestCase) -> dict:
        """评测单条"""
        result = self.rag.query(tc.question if hasattr(tc, 'question') else tc.query)

        context = result.get("context", "")
        vector_results = result.get("vector_results", [])
        enriched = result.get("enriched", [])

        # Hit: 增强上下文是否包含目标药品
        hit = tc.expected_drug[:10] in context

        # Section match: 向量 top-1 的 section 是否匹配期望
        top_section = vector_results[0].get("section", "") if vector_results else ""
        section_match = (tc.expected_section in top_section) or (tc.expected_section == "")

        return {
            "case_id": tc.case_id,
            "query": tc.query,
            "expected": tc.expected_drug,
            "difficulty": tc.difficulty,
            "category": tc.category,
            "hit": 1 if hit else 0,
            "section_match": 1 if section_match else 0,
            "top_vector_score": vector_results[0].get("score", 0) if vector_results else 0,
            "top_drug": vector_results[0].get("drug_name", "")[:20] if vector_results else "",
            "context_length": len(context),
        }

    def _mean(self, values: list) -> float:
        return sum(values) / len(values) if values else 0.0


# ═══════════════════════════════════════════
# 5. 运行入口 + 报告输出
# ═══════════════════════════════════════════

def run_full_evaluation(md_dir: str) -> dict:
    """一键运行全部评测并返回结果"""

    print("=" * 60)
    print("  药物说明书 RAG 系统 — 全面评测")
    print("=" * 60)

    # 初始化所有组件
    print("\n[Init] 加载组件...")
    parser = DrugMDParser(md_dir)
    docs = parser.parse_all()

    vs = VectorStore(persist_dir="./data/vector_store")
    kg = KnowledgeGraph()
    kg.build(docs)
    rag = HybridRAG(vs, kg, parser)

    # 生成测试集
    print(f"[Init] 文档 {len(docs)} 份，图谱 {kg.summary['nodes']} 节点 {kg.summary['edges']} 边")

    builder = TestSetBuilder(docs)
    test_cases = builder.build()
    print(f"[TestSet] 自动生成 {len(test_cases)} 条测试用例")

    easy_n = sum(1 for t in test_cases if t.difficulty == "easy")
    medium_n = sum(1 for t in test_cases if t.difficulty == "medium")
    hard_n = sum(1 for t in test_cases if t.difficulty == "hard")
    print(f"          easy={easy_n}, medium={medium_n}, hard={hard_n}")

    # ── 1. 向量检索评测 ──
    print("\n" + "-" * 40)
    print("  [1/3] 向量检索评测 (Recall@K / MRR / NDCG)")
    vec_eval = VectorEvaluator(vs)
    vec_metrics = vec_eval.evaluate(test_cases)

    print(f"  Recall@1:  {vec_metrics.get('recall@1', 0):.3f}")
    print(f"  Recall@3:  {vec_metrics.get('recall@3', 0):.3f}")
    print(f"  Recall@5:  {vec_metrics.get('recall@5', 0):.3f}")
    print(f"  MRR@3:     {vec_metrics.get('mrr@3', 0):.3f}")
    print(f"  NDCG@3:    {vec_metrics.get('ndcg@3', 0):.3f}")
    if "recall@3_easy" in vec_metrics:
        print(f"  ── 按难度 ──")
        for diff in ["easy", "medium", "hard"]:
            k = f"recall@3_{diff}"
            if k in vec_metrics:
                print(f"  Recall@3 ({diff:6s}): {vec_metrics[k]:.3f}")

    # ── 2. 知识图谱评测 ──
    print("\n" + "-" * 40)
    print("  [2/3] 知识图谱评测 (实体覆盖率 / 关系准确率)")
    kg_eval = KGEvaluator(kg, docs)
    kg_metrics = kg_eval.evaluate()

    print(f"  药品节点:  {kg_metrics['drug_coverage']}/{kg_metrics['drugs_evaluated']}")
    print(f"  适应症关系覆盖率:     {kg_metrics['has_indication_relation']:.2%}")
    print(f"  不良反应关系覆盖率:   {kg_metrics['has_side_effect_relation']:.2%}")
    print(f"  给药途径关系覆盖率:   {kg_metrics['has_route_relation']:.2%}")
    print(f"  禁忌关系覆盖率:       {kg_metrics['has_contraindication_relation']:.2%}")
    print(f"  图密度:      {kg_metrics['graph_density']:.4f}")
    print(f"  平均度数:    {kg_metrics['avg_degree']:.1f}")
    print(f"  平均每疾病治疗药品数: {kg_metrics['avg_treatments_per_disease']:.1f}")

    # ── 3. 混合RAG评测 ──
    print("\n" + "-" * 40)
    print("  [3/3] 混合RAG端到端评测 (Hit Rate / 上下文相关性)")
    rag_eval = RAGEvaluator(rag)
    rag_metrics = rag_eval.evaluate(test_cases)

    print(f"  Hit Rate (药品命中):  {rag_metrics['hit_rate']:.2%}")
    print(f"  Section Match (章节): {rag_metrics['section_match_rate']:.2%}")
    print(f"  平均向量分数:          {rag_metrics['avg_vector_score']:.3f}")
    print(f"  平均上下文长度:        {rag_metrics['avg_context_length']:.0f} chars")

    summary = {
        "vector_retrieval": vec_metrics,
        "knowledge_graph": kg_metrics,
        "hybrid_rag": rag_metrics,
        "test_cases": len(test_cases),
    }

    # 导出 JSON
    output_path = Path("./data/evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  评测完成 → {output_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"
    run_full_evaluation(md_dir)
