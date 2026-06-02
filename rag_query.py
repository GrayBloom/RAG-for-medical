"""
混合 RAG 查询引擎
结合 向量检索 (ChromaDB) + 知识图谱 (NetworkX) 实现增强检索

检索策略:
1. 向量语义检索 → 找到最相关文档片段
2. 知识图谱查询 → 找到关联实体和关系
3. 结果融合排序 → 返回综合结果
4. 可选 LLM 生成回答
"""
import re
from typing import Optional
from build_vector_store import VectorStore
from build_knowledge_graph import KnowledgeGraph
from parse_docs import DrugMDParser


class HybridRAG:
    """混合检索增强生成系统"""

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        parser: DrugMDParser,
    ):
        self.vs = vector_store
        self.kg = knowledge_graph
        self.parser = parser

    def query(
        self,
        question: str,
        top_k_vector: int = 5,
        top_k_graph: int = 3,
        fusion: bool = True,
    ) -> dict:
        """
        混合查询
        Args:
            question: 用户问题
            top_k_vector: 向量检索返回数
            top_k_graph: 图谱检索返回数
            fusion: 是否融合排序
        Returns:
            包含向量结果、图谱结果和融合结果
        """
        # 1. 向量语义检索
        vector_results = self.vs.search(question, n_results=top_k_vector)

        # 2. 知识图谱检索
        graph_results = self._kg_search(question, top_k=top_k_graph)

        # 3. 从文档中补充完整信息
        enriched = self._enrich_results(vector_results, graph_results)

        result = {
            "question": question,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "enriched": enriched,
        }

        # 4. 构造增强上下文（用于喂给 LLM）
        result["context"] = self._build_context(enriched, vector_results, graph_results)

        return result

    def _kg_search(self, question: str, top_k: int = 3) -> list[dict]:
        """知识图谱搜索"""
        results = []

        # 提取问题中的可能药品名
        drug_names = [doc.drug_name for doc in self.parser.documents]
        mentioned_drugs = [d for d in drug_names if d in question]

        # 提取问题中的关键词进行图谱实体搜索
        keywords = self._extract_keywords(question)
        entity_results = []
        for kw in keywords:
            entity_results.extend(self.kg.search_entities(kw))

        # 去重排序
        seen = set()
        for r in entity_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                results.append(r)

        results.sort(key=lambda x: x["degree"], reverse=True)

        # 对提到的药品生成子图摘要
        for drug in mentioned_drugs[:3]:
            subgraph = self.kg.query_drug(drug)
            results.append({
                "type": "drug_summary",
                "drug": drug,
                "relations": subgraph,
            })

        return results[:top_k * 2]

    def _extract_keywords(self, text: str) -> list[str]:
        """简单关键词提取 — 提取可能的疾病/症状名"""
        # 常见的查询模式关键词
        patterns = [
            r"(糖尿病|肝硬化|肝炎|脑出血|脑梗|脑卒中|心功能不全|心肌|肌萎缩)",
            r"(胆汁淤积|肾功能|肝功能|麻醉|镇静|营养|过敏)",
            r"(儿童|成人|老年|孕妇|婴儿)",
            r"(静脉|肌内|皮下|口服|注射|滴注)",
        ]
        keywords = []
        for p in patterns:
            matches = re.findall(p, text)
            keywords.extend(matches)
        return list(set(keywords))

    def _enrich_results(
        self, vector_results: list[dict], graph_results: list[dict]
    ) -> list[dict]:
        """
        将向量检索的文档与图谱实体关联
        为每个文档补充图谱中的关系信息
        """
        enriched = []
        drugs_seen = set()

        for vr in vector_results:
            drug_name = vr.get("drug_name", "")
            if drug_name not in drugs_seen:
                drugs_seen.add(drug_name)
                kg_info = self.kg.query_drug(drug_name)
                enriched.append({
                    "drug_name": drug_name,
                    "section": vr.get("section", ""),
                    "content": vr.get("content", ""),
                    "score": vr.get("score", 0),
                    "kg_relations": kg_info,
                })

        return enriched

    def _build_context(
        self,
        enriched: list[dict],
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> str:
        """构造增强上下文，用于 LLM 生成回答"""
        parts = []

        parts.append("=" * 50)
        parts.append("【知识图谱关系】")
        parts.append("=" * 50)

        for item in enriched[:3]:
            kg = item.get("kg_relations", {})
            parts.append(f"\n药品: {item['drug_name']}")
            if kg.get("treats"):
                parts.append(f"  治疗: {', '.join(kg['treats'])}")
            if kg.get("side_effects"):
                parts.append(f"  不良反应: {', '.join(kg['side_effects'][:5])}")
            if kg.get("contraindications"):
                parts.append(f"  禁忌: {', '.join(kg['contraindications'])}")
            if kg.get("interactions"):
                parts.append(f"  相互作用: {', '.join(kg['interactions'])}")
            if kg.get("routes"):
                parts.append(f"  给药途径: {', '.join(kg['routes'])}")

        parts.append("\n" + "=" * 50)
        parts.append("【相关药品说明书原文】")
        parts.append("=" * 50)

        for vr in vector_results[:5]:
            parts.append(f"\n[{vr['score']:.3f}] {vr['drug_name']} - {vr['section']}")
            parts.append(vr["content"][:500])

        return "\n".join(parts)

    def answer_with_context(
        self,
        question: str,
        max_tokens: int = 2000,
    ) -> str:
        """
        生成带上下文的回答 (不含 LLM，纯拼接上下文)
        如需 LLM 生成，可在调用方将 context 喂给任意 LLM API
        """
        result = self.query(question)
        return result["context"]


def demo():
    """演示混合检索"""
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

    print("初始化中...")

    # 加载解析器
    parser = DrugMDParser(md_dir)
    parser.parse_all()

    # 加载向量库
    vs = VectorStore(persist_dir="./data/vector_store")
    print(f"  向量库: {vs.get_collection().count()} 条记录")

    # 加载知识图谱
    kg = KnowledgeGraph()
    kg.build(parser.documents)
    print(f"  知识图谱: {kg.summary['nodes']} 节点, {kg.summary['edges']} 边")

    # 创建混合RAG
    rag = HybridRAG(vs, kg, parser)

    # 测试查询
    questions = [
        "糖尿病有哪些治疗药物？",
        "丙泊酚的麻醉诱导剂量和注意事项？",
        "肝功能不全患者用什么药需要注意什么？",
        "儿童使用全身麻醉药的剂量？",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print(f"❓ {q}")
        print("=" * 60)

        result = rag.query(q)

        print("\n📊 向量语义检索 Top-3:")
        for vr in result["vector_results"][:3]:
            print(f"  [{vr['score']:.3f}] {vr['drug_name']} | {vr['section']}")

        print("\n🧠 知识图谱实体:")
        for gr in result["graph_results"][:5]:
            if isinstance(gr, dict) and "label" in gr:
                print(f"  [{gr.get('type', '?')}] {gr['label']}")

        print("\n🔗 融合结果:")
        for en in result["enriched"][:3]:
            kg_rel = en.get("kg_relations", {})
            print(f"  📄 {en['drug_name']} (score={en['score']:.3f})")
            if kg_rel.get("treats"):
                print(f"     ↳ 治疗: {', '.join(kg_rel['treats'][:2])}")
            if kg_rel.get("routes"):
                print(f"     ↳ 给药: {', '.join(kg_rel['routes'])}")
            if kg_rel.get("contraindications"):
                print(f"     ↳ 禁忌: {', '.join(kg_rel['contraindications'][:2])}")


if __name__ == "__main__":
    demo()
