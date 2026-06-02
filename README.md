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
