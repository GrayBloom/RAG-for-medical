#!/bin/bash
# ============================================
#  LLM 评测配置 — 修改参数后 source 执行
#  用法: source config.sh
# ============================================

# ── LLM API 地址 (OpenAI 兼容格式) ──
export EVAL_BASE_URL="https://token-plan-cn.xiaomimimo.com/anthropic"

# ── API Key (默认读取 Hermes 的 xiaomi 密钥) ──
if [ -z "$EVAL_API_KEY" ]; then
    if [ -n "$XIAOMI_API_KEY" ]; then
        export EVAL_API_KEY="$XIAOMI_API_KEY"
    else
        echo "[WARN] 请先设置 EVAL_API_KEY 或 XIAOMI_API_KEY"
    fi
fi

# ── 评测用模型（打分用，建议用小模型省成本） ──
export EVAL_MODEL="mimo-v2.5-pro"

# ── RAG 系统数据路径 ──
export MD_DATA_DIR="/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

# ═══════════════════════════════════════════
#  备选配置（取消注释后生效）
# ═══════════════════════════════════════════

# ── OpenAI ──
# export EVAL_BASE_URL="https://api.openai.com/v1"
# export EVAL_API_KEY="sk-xxx"
# export EVAL_MODEL="gpt-4o-mini"

# ── DeepSeek ──
# export EVAL_BASE_URL="https://api.deepseek.com/v1"
# export EVAL_API_KEY="sk-xxx"
# export EVAL_MODEL="deepseek-chat"

# ── 本地 Ollama ──
# export EVAL_BASE_URL="http://localhost:11434/v1"
# export EVAL_API_KEY="ollama"
# export EVAL_MODEL="qwen2.5:7b"

echo "===== LLM 评测配置 ====="
echo "  BASE_URL: ${EVAL_BASE_URL}"
echo "  MODEL:    ${EVAL_MODEL}"
echo "  RAG数据:  ${MD_DATA_DIR}"
echo "========================="
