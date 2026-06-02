"""
LLM 生成质量评测模块
维度：忠实度 / 相关性 / 完整性 / 声明支持率

两种模式:
  LLM模式   — 用 LLM-as-Judge 打分 (忠实度/相关性/完整性)
  无LLM模式  — 纯向量验证 (仅声明支持率)

依赖: requests (标准库无此依赖, 改用 urllib 或要求安装)
"""
import os
import re
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

from parse_docs import DrugMDParser
from build_vector_store import VectorStore
from build_knowledge_graph import KnowledgeGraph
from rag_query import HybridRAG
from evaluate import TestSetBuilder, TestCase


# ═══════════════════════════════════════════
# 0. 轻量级 LLM 客户端 (OpenAI 兼容, 无额外依赖)
# ═══════════════════════════════════════════

class LLMClient:
    """OpenAI-compatible API 客户端，纯 stdlib"""

    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.base_url = (base_url or os.getenv("EVAL_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("EVAL_API_KEY", "")
        self.model = model or os.getenv("EVAL_MODEL", "mimo-v2.5-pro")

        if not self.api_key:
            raise ValueError("未设置 API Key，请先执行: source config.sh  或 export EVAL_API_KEY=...")

        # 自动补齐 /chat/completions
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"

    def chat(self, messages: list[dict], max_tokens: int = 512,
             temperature: float = 0.0) -> str:
        """发请求并返回文本"""
        import urllib.request
        import urllib.error

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(self.base_url, data=body, headers=headers, method="POST")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else ""
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return f"[HTTP {e.code}] {body[:200]}"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return f"[Error] {str(e)[:200]}"

        return "[Error] max retries exceeded"


# ═══════════════════════════════════════════
# 1. RAG 答案生成器
# ═══════════════════════════════════════════

class RAGAnswerGenerator:
    """用 RAG 上下文 + LLM 生成答案"""

    SYSTEM_PROMPT = """你是一个药品信息助手。请根据以下提供的药品说明书上下文回答问题。
规则:
1. 只使用上下文中的信息，不要编造
2. 如果上下文中没有相关信息，直接说"说明书中未提及"
3. 回答简洁准确，用中文
4. 涉及剂量等数字时严格复述原文"""

    def __init__(self, rag: HybridRAG, llm: Optional[LLMClient] = None):
        self.rag = rag
        self.llm = llm

    def generate(self, question: str) -> dict:
        """生成答案，返回答案和使用的上下文"""
        rag_result = self.rag.query(question)
        context = rag_result.get("context", "")

        if self.llm:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"上下文:\n{context}\n\n问题: {question}"},
            ]
            answer = self.llm.chat(messages, max_tokens=1024)
        else:
            # 无 LLM 时用向量检索结果拼成答案（用于验证声明支持率）
            vr = rag_result.get("vector_results", [])
            if vr:
                texts = [r["content"][:200] for r in vr[:3]]
                answer = "\n".join(texts)
            else:
                answer = "未找到相关内容"

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "context_length": len(context),
            "vector_results": rag_result.get("vector_results", []),
        }


# ═══════════════════════════════════════════
# 2. LLM-as-Judge 评分器
# ═══════════════════════════════════════════

class LLMJudge:
    """用 LLM 对生成答案评分 (忠实度 / 相关性 / 完整性)"""

    FAITHFULNESS_PROMPT = """你的任务是评估一个回答是否**忠实于给定的上下文**（不编造、不歪曲）。

评分标准 (1-5):
5 — 完全忠实，所有陈述都能在上下文中找到依据
4 — 基本忠实，个别细节有轻微出入
3 — 部分忠实，有一些与上下文不一致的地方
2 — 明显不忠实，多处偏离上下文
1 — 完全不忠实，大部分内容没有依据

上下文:
{context}

问题: {question}

回答: {answer}

请输出 JSON: {{"score": <1-5>, "reason": "<一句中文理由>"}}
只输出 JSON，不要其他内容。"""

    RELEVANCE_PROMPT = """你的任务是评估一个回答是否**直接回应了问题**。

评分标准 (1-5):
5 — 完美回应，直接命中问题核心
4 — 回应了大部分问题，略微跑题
3 — 回应了部分问题，但遗漏关键点
2 — 回答与问题相关度低
1 — 完全答非所问

问题: {question}

回答: {answer}

请输出 JSON: {{"score": <1-5>, "reason": "<一句中文理由>"}}
只输出 JSON，不要其他内容。"""

    COMPLETENESS_PROMPT = """你的任务是评估一个回答是否**完整覆盖了上下文中的关键信息**。

评分标准 (1-5):
5 — 完整覆盖了上下文中所有相关关键信息
4 — 覆盖了大部分关键信息，有少量遗漏
3 — 覆盖了一半左右的关键信息
2 — 遗漏了大量关键信息
1 — 几乎没有覆盖关键信息

上下文:
{context}

问题: {question}

回答: {answer}

请输出 JSON: {{"score": <1-5>, "reason": "<一句中文理由>"}}
只输出 JSON，不要其他内容。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """三项评分"""
        ctx_trimmed = context[:3000]  # 截断，省 token

        scores = {}

        # 忠实度
        scores["faithfulness"] = self._score(
            self.FAITHFULNESS_PROMPT.format(question=question, answer=answer, context=ctx_trimmed)
        )

        # 相关性
        scores["relevance"] = self._score(
            self.RELEVANCE_PROMPT.format(question=question, answer=answer)
        )

        # 完整性
        scores["completeness"] = self._score(
            self.COMPLETENESS_PROMPT.format(question=question, answer=answer, context=ctx_trimmed)
        )

        # 综合分
        scores["overall"] = (
            scores["faithfulness"]["score"] * 0.4
            + scores["relevance"]["score"] * 0.3
            + scores["completeness"]["score"] * 0.3
        )
        scores["overall"] = round(scores["overall"], 1)

        return scores

    def _score(self, prompt: str) -> dict:
        """调用 LLM 打分并解析 JSON"""
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm.chat(messages, max_tokens=256, temperature=0.0)

        # 解析 JSON
        try:
            # 提取 JSON 块
            match = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {"score": int(data["score"]), "reason": data.get("reason", "")}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # fallback: 尝试从文本中提取数字
        nums = re.findall(r'[1-5]', raw)
        score = int(nums[0]) if nums else 3
        return {"score": score, "reason": raw[:100].replace("\n", " ")}


# ═══════════════════════════════════════════
# 3. 声明验证器 (纯向量, 无需 LLM)
# ═══════════════════════════════════════════

class ClaimVerifier:
    """逐句验证：答案中每句话是否能在向量库中找到支持的证据"""

    def __init__(self, vector_store: VectorStore, threshold: float = 0.55):
        self.vs = vector_store
        self.threshold = threshold

    def verify(self, answer: str, context: str = "") -> dict:
        """验证答案中每条声明是否在上下文中找到支持。
        策略: 文本子串匹配 (声明应直接来自上下文) + 向量相似度兜底
        """
        claims = self._split_claims(answer)
        if not claims:
            return {"supported": 0, "total": 0, "support_rate": 1.0, "details": []}

        details = []
        supported = 0

        for claim in claims:
            # 方法1: 检查 claim 中的关键词是否出现在 context 中
            evidence_text, overlap_score = self._find_best_match(claim, context)

            # 方法2: 向量兜底 (如果文本匹配失败)
            if overlap_score < 0.15:
                results = self.vs.search(claim, n_results=1)
                vec_score = results[0]["score"] if results else 0
                if vec_score >= self.threshold:
                    overlap_score = vec_score
                    evidence_text = results[0]["content"][:120] if results else ""

            is_supported = overlap_score >= 0.15
            if is_supported:
                supported += 1

            details.append({
                "claim": claim,
                "supported": is_supported,
                "evidence_score": round(overlap_score, 3),
                "evidence": evidence_text[:120],
            })

        return {
            "supported": supported,
            "total": len(claims),
            "support_rate": supported / len(claims),
            "details": details,
        }

    def _find_best_match(self, claim: str, context: str) -> tuple:
        """在上下文中找最佳子串匹配，返回 (匹配文本, 匹配率)"""
        if not context:
            return "", 0.0

        # 将 claim 拆成词级别的片段去匹配
        # 按标点/空格把 claim 再拆一次
        tokens = re.split(r'[，,、\s:：]+', claim)
        tokens = [t for t in tokens if len(t) >= 2]

        if not tokens:
            return "", 0.0

        matched = sum(1 for t in tokens if t in context)
        score = matched / len(tokens)
        return context[:120], score

    def _split_claims(self, text: str) -> list[str]:
        """把答案拆成独立声明 (按句号/分号/换行)"""
        # 只移除行首 markdown 标记（*, #, >），不碰正文中的连字符
        text = re.sub(r'^[*#>]+\s*', '', text, flags=re.MULTILINE)
        # 按标点拆分
        raw = re.split(r'[。；;.\n]+', text)
        claims = []
        for s in raw:
            s = s.strip()
            if len(s) >= 6 and not re.match(r'^(可以|是的|好的|以上|综上|根据)', s):
                claims.append(s)
        return claims


# ═══════════════════════════════════════════
# 4. 生成评测主控
# ═══════════════════════════════════════════

class GenerationEvaluator:
    """生成质量评测主控"""

    def __init__(self, rag: HybridRAG, vector_store: VectorStore,
                 llm: Optional[LLMClient] = None):
        self.generator = RAGAnswerGenerator(rag, llm)
        self.judge = LLMJudge(llm) if llm else None
        self.verifier = ClaimVerifier(vector_store)

    def evaluate(self, test_cases: list[TestCase], sample_size: int = None) -> dict:
        """
        运行评测流程:
        1. 对每个测试用例 → RAG生成答案
        2. LLM-as-Judge 打分 (如有 LLM)
        3. 声明验证 (纯向量)
        """
        if sample_size:
            import random
            test_cases = random.sample(test_cases, min(sample_size, len(test_cases)))

        results = []
        print(f"评测 {len(test_cases)} 条用例...")

        for i, tc in enumerate(test_cases):
            query = tc.query
            print(f"  [{i+1}/{len(test_cases)}] {query[:40]}...", end=" ", flush=True)

            # Step 1: 生成答案
            gen = self.generator.generate(query)
            answer = gen["answer"]
            context = gen["context"]

            # Step 2: LLM 评分 (可选)
            judge_scores = {}
            if self.judge:
                judge_scores = self.judge.evaluate(query, answer, context)
                print(f"LLM={judge_scores.get('overall', '?')}", end=" ", flush=True)

            # Step 3: 声明验证
            claim_result = self.verifier.verify(answer, context=context)
            print(f"Support={claim_result['support_rate']:.0%}")

            results.append({
                "case_id": tc.case_id,
                "query": query,
                "expected_drug": tc.expected_drug,
                "difficulty": tc.difficulty,
                "answer": answer[:500],
                "context_length": gen["context_length"],
                "judge_scores": judge_scores,
                "claim_verification": {
                    "support_rate": claim_result["support_rate"],
                    "supported": claim_result["supported"],
                    "total": claim_result["total"],
                },
            })

        # 汇总
        summary = self._summarize(results)
        return {"summary": summary, "details": results}

    def _summarize(self, results: list[dict]) -> dict:
        """汇总统计"""
        n = len(results)

        summary = {"total_cases": n}

        # LLM 评分汇总
        if results and results[0].get("judge_scores"):
            for metric in ["faithfulness", "relevance", "completeness"]:
                vals = []
                for r in results:
                    js = r.get("judge_scores", {})
                    item = js.get(metric, {})
                    if isinstance(item, dict):
                        vals.append(item.get("score", 0))
                    elif isinstance(item, (int, float)):
                        vals.append(item)
                if vals:
                    summary[f"avg_{metric}"] = round(sum(vals) / len(vals), 2)

            # overall 是 float，单独处理
            overalls = [
                r["judge_scores"]["overall"]
                for r in results
                if r.get("judge_scores") and isinstance(r["judge_scores"].get("overall"), (int, float))
            ]
            if overalls:
                summary["avg_overall"] = round(sum(overalls) / len(overalls), 2)

        # 声明支持率汇总
        support_rates = [
            r["claim_verification"]["support_rate"]
            for r in results
        ]
        summary["avg_claim_support_rate"] = round(sum(support_rates) / len(support_rates), 3)

        # 按难度拆分
        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)
        for diff, items in by_diff.items():
            if items[0].get("judge_scores"):
                vals = []
                for it in items:
                    js = it.get("judge_scores", {})
                    ov = js.get("overall", 0)
                    vals.append(ov if isinstance(ov, (int, float)) else 0)
                if vals:
                    summary[f"avg_overall_{diff}"] = round(sum(vals) / len(vals), 2)
            sr = [it["claim_verification"]["support_rate"] for it in items]
            summary[f"support_rate_{diff}"] = round(sum(sr) / len(sr), 3)

        return summary


# ═══════════════════════════════════════════
# 5. 运行入口
# ═══════════════════════════════════════════

def run_generation_eval(md_dir: str, llm_mode: bool = True, sample_size: int = None):
    """
    运行生成评测
    Args:
        md_dir: MD 数据目录
        llm_mode: 是否启用 LLM 评分 (False 则只做声明验证)
        sample_size: 限制评测条数
    """
    print("=" * 60)
    print("  RAG 生成质量评测")
    print("=" * 60)

    # 初始化 LLM
    llm = None
    if llm_mode:
        try:
            llm = LLMClient()
            print(f"\n[LLM] {llm.model} @ {llm.base_url}")
        except ValueError as e:
            print(f"\n[WARN] {e}")
            print("[INFO] 降级为纯向量模式（仅声明验证）")
            llm_mode = False

    # 初始化 RAG 组件
    print("\n[Init] 加载 RAG 系统...")
    parser = DrugMDParser(md_dir)
    docs = parser.parse_all()

    vs = VectorStore(persist_dir="./data/vector_store")
    kg = KnowledgeGraph()
    kg.build(docs)
    rag = HybridRAG(vs, kg, parser)

    # 生成测试集
    builder = TestSetBuilder(docs)
    all_cases = builder.build()

    # 按难度分层采样
    if sample_size and sample_size < len(all_cases):
        import random
        random.seed(42)
        cases_by_diff = defaultdict(list)
        for c in all_cases:
            cases_by_diff[c.difficulty].append(c)
        sampled = []
        for diff in ["easy", "medium", "hard"]:
            n = max(1, int(sample_size * len(cases_by_diff[diff]) / len(all_cases)))
            sampled.extend(random.sample(cases_by_diff[diff], min(n, len(cases_by_diff[diff]))))
        test_cases = sampled
    else:
        test_cases = all_cases

    print(f"[TestSet] {len(test_cases)} 条用例 (总计 {len(all_cases)})")

    # 运行评测
    evaluator = GenerationEvaluator(rag, vs, llm)
    result = evaluator.evaluate(test_cases)

    # 输出报告
    print("\n" + "=" * 60)
    print("  评测结果")
    print("=" * 60)

    s = result["summary"]

    print(f"\n  用例数: {s['total_cases']}")

    if s.get("avg_overall"):
        print(f"\n  ── LLM-as-Judge 评分 (1-5) ──")
        print(f"  忠实度 (Faithfulness):  {s.get('avg_faithfulness', 0):.2f}")
        print(f"  相关性 (Relevance):     {s.get('avg_relevance', 0):.2f}")
        print(f"  完整性 (Completeness):  {s.get('avg_completeness', 0):.2f}")
        print(f"  综合分 (Overall):       {s.get('avg_overall', 0):.2f}")

    print(f"\n  ── 声明支持率 (向量验证) ──")
    print(f"  支持率:                 {s.get('avg_claim_support_rate', 0):.1%}")

    # 按难度
    diffs_present = [d for d in ["easy", "medium", "hard"] if f"avg_overall_{d}" in s]
    if diffs_present:
        print(f"\n  ── 按难度拆分 ──")
        for diff in diffs_present:
            ov = s.get(f"avg_overall_{diff}", "")
            sr = s.get(f"support_rate_{diff}", "")
            parts = []
            if ov:
                parts.append(f"Overall={ov}")
            if sr:
                parts.append(f"Support={sr:.0%}")
            print(f"  {diff:6s}: {', '.join(parts)}")

    # 导出
    output_path = Path("./data/evaluation_generation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {output_path}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "MD_DATA_DIR", "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"
    )
    llm_mode = "--no-llm" not in sys.argv
    sample_size = None
    for i, arg in enumerate(sys.argv):
        if arg == "--sample" and i + 1 < len(sys.argv):
            sample_size = int(sys.argv[i + 1])

    run_generation_eval(md_dir, llm_mode=llm_mode, sample_size=sample_size)
