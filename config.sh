#!/bin/bash
# ============================================
#  LLM 评测配置 — 生成和评测用两个独立 LLM
#  用法: source config.sh
# ============================================

# ╔══════════════════════════════════════════════════════╗
# ║            ★ 手动设置 API Key (在此粘贴) ★            ║
# ╚══════════════════════════════════════════════════════╝

# ── 生成 LLM (写答案) ──
export EVAL_GEN_API_KEY="***"   # ← 粘贴生成模型的 API Key

# ── 评测 LLM (打分)   ──
export EVAL_JUDGE_API_KEY="***" # ← 粘贴评测模型的 API Key

# ═══════════════════════════════════════════════════════
#  以下为地址和模型配置（一般不需要改）
# ═══════════════════════════════════════════════════════

# ── 生成 LLM ──
export EVAL_GEN_BASE_URL="https://token-plan-cn.xiaomimimo.com/anthropic"
export EVAL_GEN_MODEL="mimo-v2.5-pro"

# ── 评测 LLM ──
export EVAL_JUDGE_BASE_URL="https://api.deepseek.com/v1"
export EVAL_JUDGE_MODEL="deepseek-chat"

# ── RAG 数据路径 ──
export MD_DATA_DIR="/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

# ── 兜底: 如果上面还是占位符，自动读 Hermes 的已有的 Key ──
if [ "$EVAL_GEN_API_KEY" = "***" ] && [ -n "$XIAOMI_API_KEY" ]; then
    export EVAL_GEN_API_KEY="$XIA..."
fi
if [ "$EVAL_JUDGE_API_KEY" = "***" ] && [ -n "$DEEPSEEK_API_KEY" ]; then
    export EVAL_JUDGE_API_KEY="$DEE..."
fi

# ═══════════════════════════════════════════════════════
#  备选 (取消注释后生效)
# ═══════════════════════════════════════════════════════

# ── 懒人: 生成和评测用同一个 ──
# export EVAL_JUDGE_BASE_URL="$EVAL_GEN_BASE_URL"
# export EVAL_JUDGE_MODEL="$EVAL_GEN_MODEL"
# export EVAL_JUDGE_API_KEY="$EVAL_GEN_API_KEY"

# ── DeepSeek ──
# export EVAL_GEN_BASE_URL="https://api.deepseek.com/v1"
# export EVAL_GEN_MODEL="deepseek-chat"
# export EVAL_GEN_API_KEY="***"
# export EVAL_JUDGE_BASE_URL="https://api.deepseek.com/v1"
# export EVAL_JUDGE_MODEL="deepseek-chat"
# export EVAL_JUDGE_API_KEY="***"

# ── OpenAI ──
# export EVAL_GEN_BASE_URL="https://api.openai.com/v1"
# export EVAL_GEN_MODEL="gpt-4o"
# export EVAL_GEN_API_KEY="***"
# export EVAL_JUDGE_BASE_URL="https://api.openai.com/v1"
# export EVAL_JUDGE_MODEL="gpt-4o-mini"
# export EVAL_JUDGE_API_KEY="***"

# ── 本地 Ollama ──
# export EVAL_GEN_BASE_URL="http://localhost:11434/v1"
# export EVAL_GEN_MODEL="qwen2.5:14b"
# export EVAL_GEN_API_KEY="ollama"
# export EVAL_JUDGE_BASE_URL="http://localhost:11434/v1"
# export EVAL_JUDGE_MODEL="qwen2.5:7b"
# export EVAL_JUDGE_API_KEY="ollama"

echo "===== LLM 评测配置 ====="
echo "  生成 (写答案): ${EVAL_GEN_MODEL} @ ${EVAL_GEN_BASE_URL}"
echo "  评测 (打分):   ${EVAL_JUDGE_MODEL} @ ${EVAL_JUDGE_BASE_URL}"
echo "  RAG数据:       ${MD_DATA_DIR}"
echo "===== Key 检测 ====="
[ -n "$EVAL_GEN_API_KEY" ] && echo "  生成 Key: ✅ (${#EVAL_GEN_API_KEY} 字符)" || echo "  生成 Key: ❌ 未设置"
[ -n "$EVAL_JUDGE_API_KEY" ] && echo "  评测 Key: ✅ (${#EVAL_JUDGE_API_KEY} 字符)" || echo "  评测 Key: ❌ 未设置"
echo "========================="
