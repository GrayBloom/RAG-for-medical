#!/bin/bash
# ============================================
#  LLM 评测配置 — 生成和评测用两个独立 LLM
#  用法: source config.sh
# ============================================

# ═══════════════════════════════════════════
#  1. 生成 LLM — 用来写答案（建议能力强）
# ═══════════════════════════════════════════
export EVAL_GEN_BASE_URL="https://token-plan-cn.xiaomimimo.com/anthropic"
export EVAL_GEN_MODEL="mimo-v2.5-pro"

# ═══════════════════════════════════════════
#  2. 评测 LLM — 用来打分（建议便宜/中立）
# ═══════════════════════════════════════════
export EVAL_JUDGE_BASE_URL="https://api.deepseek.com/v1"
export EVAL_JUDGE_MODEL="deepseek-chat"

# ═══════════════════════════════════════════
#  API Key (共用一组，或分别设置)
# ═══════════════════════════════════════════
if [ -z "$EVAL_GEN_API_KEY" ]; then
    if [ -n "$XIAOMI_API_KEY" ]; then
        export EVAL_GEN_API_KEY="$XIAOMI_API_KEY"
    fi
fi
if [ -z "$EVAL_JUDGE_API_KEY" ]; then
    if [ -n "$DEEPSEEK_API_KEY" ]; then
        export EVAL_JUDGE_API_KEY="$DEEPSEEK_API_KEY"
    fi
fi

# ── RAG 系统数据路径 ──
export MD_DATA_DIR="/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

# ═══════════════════════════════════════════
#  备选配置（取消注释后生效）
# ═══════════════════════════════════════════

# ── 生成: OpenAI ──
# export EVAL_GEN_BASE_URL="https://api.openai.com/v1"
# export EVAL_GEN_MODEL="gpt-4o"

# ── 评测: OpenAI 便宜模型 ──
# export EVAL_JUDGE_BASE_URL="https://api.openai.com/v1"
# export EVAL_JUDGE_MODEL="gpt-4o-mini"

# ── 生成: 本地 Ollama ──
# export EVAL_GEN_BASE_URL="http://localhost:11434/v1"
# export EVAL_GEN_MODEL="qwen2.5:14b"

# ── 评测: 本地 Ollama 小模型 ──
# export EVAL_JUDGE_BASE_URL="http://localhost:11434/v1"
# export EVAL_JUDGE_MODEL="qwen2.5:7b"

# ── 懒人模式: 生成和评测用同一个模型 ──
# export EVAL_JUDGE_BASE_URL="$EVAL_GEN_BASE_URL"
# export EVAL_JUDGE_MODEL="$EVAL_GEN_MODEL"
# export EVAL_JUDGE_API_KEY="$EVAL_GEN_API_KEY"

echo "===== LLM 评测配置 ====="
echo "  生成 (写答案): ${EVAL_GEN_MODEL} @ ${EVAL_GEN_BASE_URL}"
echo "  评测 (打分):   ${EVAL_JUDGE_MODEL} @ ${EVAL_JUDGE_BASE_URL}"
echo "  RAG数据:       ${MD_DATA_DIR}"
echo "========================="
