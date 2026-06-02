# 药物说明书 知识图谱 + 向量知识库 RAG 系统

## 架构

```
                    ┌─────────────────┐
                    │   用户查询        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐
   │  向量语义检索      │          │  知识图谱查询      │
   │  ChromaDB + BGE   │          │  NetworkX 图      │
   │  (语义匹配)        │          │  (实体+关系)       │
   └────────┬─────────┘          └────────┬─────────┘
            │                             │
            └──────────────┬──────────────┘
                           ▼
              ┌─────────────────────┐
              │   结果融合 & 排序     │
              │   HybridRAG Engine  │
              └─────────────────────┘
```

## 项目结构

```
rag-system/
├── parse_docs.py              # MD 文档解析器 (9份说明书 → 97个切片)
├── build_vector_store.py      # 向量库构建 (ChromaDB + BGE embedding)
├── build_knowledge_graph.py   # 知识图谱构建 (NetworkX)
├── rag_query.py               # 混合 RAG 查询引擎
├── evaluate.py                # 评测模块（自动生成测试集）
├── build_all.py               # 一键构建脚本
├── requirements.txt
├── data/
│   ├── vector_store/          # ChromaDB 持久化
│   ├── knowledge_graph.json   # 图谱 JSON
│   ├── knowledge_graph.html   # 交互式可视化
│   └── models/                # 本地 embedding 模型
└── .venv/                     # Python 虚拟环境
```

## 快速开始

### 1. 安装依赖

```bash
cd rag-system
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt modelscope
```

### 2. 一键构建

```bash
python build_all.py "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"
```

构建内容：
- 解析 9 份说明书 → 97 个文本切片
- 生成 512 维 BGE 中文向量，存入 ChromaDB
- 抽取 70 个实体节点 + 80 条关系边

### 3. 查询

```python
from rag_query import HybridRAG
from build_vector_store import VectorStore
from build_knowledge_graph import KnowledgeGraph
from parse_docs import DrugMDParser

# 初始化
parser = DrugMDParser("path/to/md")
parser.parse_all()
vs = VectorStore(persist_dir="./data/vector_store")
kg = KnowledgeGraph()
kg.build(parser.documents)
rag = HybridRAG(vs, kg, parser)

# 查询
result = rag.query("糖尿病有哪些治疗药物？")
print(result["context"])  # 增强上下文，可喂给任何 LLM
```

### 4. 查看知识图谱可视化

浏览器打开 `data/knowledge_graph.html`

## 评测

`evaluate.py` 从结构化字段**自动生成测试集**（无需人工标注），覆盖三个维度：

```bash
python evaluate.py "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"
# 报告输出到 data/evaluation_report.json
```

### 评测维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **向量检索** | Recall@K, MRR, NDCG | 语义搜索能否把正确答案排在前列 |
| **知识图谱** | 实体覆盖率, 关系准确率 | 从说明书抽取的实体和关系是否完整 |
| **混合RAG** | Hit Rate, 上下文相关性 | 向量+图谱融合后的端到端质量 |

> 测试用例自动生成策略：
> - **easy** — 查询中含药品名 ("介绍一下丙泊酚")
> - **medium** — 查询含适应症/禁忌关键词 ("什么药治疗糖尿病")
> - **hard** — 纯症状描述不含药品名 ("患者出现肝内胆汁淤积如何处理")

### 当前评测结果

```
评测用例: 71 条 (easy=36, medium=29, hard=6)

── 向量检索 ──
  Recall@1:  0.915     ← 91.5% 查询第一位就是正确答案
  Recall@5:  0.944     ← 前5位覆盖 94.4%
  MRR@3:     0.922     ← 正确答案平均排在第 1.08 位
  NDCG@3:    0.932
  Recall@3 (easy):   1.000
  Recall@3 (medium): 0.862
  Recall@3 (hard):   0.667

── 知识图谱 ──
  适应症关系覆盖率:     100%
  不良反应关系覆盖率:   100%
  禁忌关系覆盖率:       100%
  给药途径关系覆盖率:   77.8%
  图密度 / 平均度数:    0.017 / 2.3

── 混合RAG ──
  Hit Rate (药品命中):  94.4%
  平均向量分数:          0.807
  平均上下文长度:        1090 chars
```

## 数据统计

| 指标 | 数值 |
|------|------|
| 药品说明书 | 9 份 |
| 文本切片 | 97 个 |
| 图谱节点 | 70 |
| 图谱边 | 80 |
| 实体类型 | Drug, Disease, Symptom, Route, Population |
| 关系类型 | TREATS, HAS_SIDE_EFFECT, CONTRAINDICATED_FOR, INTERACTS_WITH, ADMINISTERED_VIA |

## 查询示例

```
❓ 糖尿病有哪些治疗药物？
→ 30/70混合重组人胰岛素注射液 (score=0.677)
  ↳ 治疗: 糖尿病

❓ 丙泊酚的麻醉诱导剂量？
→ 丙泊酚中／长链脂肪乳注射液
  ↳ 用法用量: 成人诱导剂量 1.5~2.5 mg/kg
  ↳ 给药途径: 静脉给药

❓ 肝功能不全患者用什么药需要注意什么？
→ 丙氨酰谷氨酰胺注射液
  ↳ 禁忌: 肝功能不全患者禁用
→ 丁二磺酸腺苷蛋氨酸肠溶片
  ↳ 治疗: 肝硬化, 胆汁淤积
```
