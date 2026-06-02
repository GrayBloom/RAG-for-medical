"""
声明验证逻辑测试（纯 Python，不依赖 torch/venv）
"""
import re

# ── 直接复制 ClaimVerifier 的核心逻辑 ──

def split_claims(text: str) -> list[str]:
    """把答案拆成独立声明"""
    text = re.sub(r'^[*#>]+\s*', '', text, flags=re.MULTILINE)
    raw = re.split(r'[。；;.\n]+', text)
    claims = []
    for s in raw:
        s = s.strip()
        if len(s) >= 6 and not re.match(r'^(可以|是的|好的|以上|综上|根据)', s):
            claims.append(s)
    return claims

def find_best_match(claim: str, context: str) -> tuple:
    """在上下文中找最佳子串匹配"""
    if not context:
        return "", 0.0
    tokens = re.split(r'[，,、\s:：]+', claim)
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return "", 0.0
    matched = sum(1 for t in tokens if t in context)
    score = matched / len(tokens)
    return context[:120], score

def verify(answer: str, context: str = "", threshold: float = 0.15) -> dict:
    claims = split_claims(answer)
    if not claims:
        return {"supported": 0, "total": 0, "support_rate": 1.0, "details": []}

    details = []
    supported = 0
    for claim in claims:
        evidence_text, overlap_score = find_best_match(claim, context)
        is_supported = overlap_score >= threshold
        if is_supported:
            supported += 1
        details.append({
            "claim": claim,
            "supported": is_supported,
            "evidence_score": round(overlap_score, 3),
            "evidence": evidence_text[:120],
        })
    return {
        "supported": supported, "total": len(claims),
        "support_rate": supported / len(claims), "details": details,
    }


# ── 测试 ──

context = """
药品名称：30/70混合重组人胰岛素注射液
适应症：1型或2型糖尿病。
用法用量：使用前检查笔芯规格。准备胰岛素：将笔芯装入注射笔，安装针头，
颠倒8-10次混匀药液，调节剂量，排尽气泡后注射。
注射部位：选择上臂、大腿、臀部或腹部。
"""

# 测试1: 回答全部来自上下文
answer1 = """根据说明书，30/70混合重组人胰岛素注射液的用法如下：
使用前检查笔芯规格。将笔芯装入注射笔，安装针头。
颠倒8-10次混匀药液。调节剂量后排尽气泡后注射。
注射部位可选择上臂、大腿、臀部或腹部。"""

r1 = verify(answer1, context)
print(f"测试1 (忠实回答): 支持率 = {r1['support_rate']:.0%} (期望: ≥50%)")
for d in r1['details'][:5]:
    print(f"  [{'✓' if d['supported'] else '✗'}] {d['claim'][:50]}...")

# 测试2: 含明显幻觉
answer2 = """根据说明书，30/70混合重组人胰岛素注射液每天吃三次，
每次饭后服用。不需要任何准备。副作用包括脱发和失眠。"""

r2 = verify(answer2, context)
print(f"\n测试2 (幻觉回答): 支持率 = {r2['support_rate']:.0%} (期望: <30%)")
for d in r2['details']:
    print(f"  [{'✓' if d['supported'] else '✗'}] {d['claim'][:50]}...")

# 测试3: 空上下文
r3 = verify(answer1, context="")
print(f"\n测试3 (空上下文): 支持率 = {r3['support_rate']:.0%} (期望: 0%)")
