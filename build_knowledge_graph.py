"""
知识图谱构建模块
从药物说明书中抽取实体和关系，构建 NetworkX 图
支持导出为 JSON 和 PyVis HTML 可视化
"""
import re
import json
import jieba
from pathlib import Path
from collections import defaultdict
import networkx as nx
from parse_docs import DrugMDParser, DrugDocument


class EntityExtractor:
    """从药物说明书文本中抽取实体"""

    # 疾病/症状关键词模式
    DISEASE_PATTERNS = [
        r"糖尿病", r"肝硬化", r"肝炎", r"脑出血", r"脑卒中", r"脑梗",
        r"心功能不全", r"心肌疾患", r"心肌梗死", r"肌萎缩", r"胆汁淤积",
        r"肾功能不全", r"肝功能不全", r"高胱氨酸尿", r"高同型半胱氨酸血症",
        r"血清素综合征", r"高甘油三酯血症", r"房窦综合征", r"窦房结功能不全",
        r"缺血性脑卒中", r"妊娠期肝内胆汁淤积",
    ]

    # 给药的途径
    ROUTE_PATTERNS = [
        "静脉注射", "静脉滴注", "肌内注射", "肌肉注射", "皮下注射",
        "口服", "肠外营养", "静脉给药", "靶控输注",
    ]

    # 人群
    POPULATION_PATTERNS = [
        r"成人", r"儿童", r"老年人?", r"孕妇", r"哺乳期",
        r"婴儿", r"新生儿", r"重症监护患者",
    ]

    # 药物相互作用
    DRUG_INTERACTION_DRUGS = [
        "氯米帕明", "利多卡因", "口服避孕药", "肾上腺皮质激素",
        "甲状腺激素", "水杨酸", "磺胺", "5-羟色胺再摄取抑制剂",
        "SSRIs", "三环抗抑郁剂",
    ]

    def extract_diseases(self, text: str) -> list[str]:
        """从文本中提取疾病/适应症"""
        found = []
        for pattern in self.DISEASE_PATTERNS:
            if re.search(pattern, text):
                found.append(pattern.replace("?", ""))
        return found

    def extract_routes(self, text: str) -> list[str]:
        """提取给药途径"""
        found = []
        for route in self.ROUTE_PATTERNS:
            if route in text:
                found.append(route)
        return found

    def extract_populations(self, text: str) -> list[str]:
        """提取适用人群"""
        found = []
        for pattern in self.POPULATION_PATTERNS:
            if re.search(pattern, text):
                clean = pattern.replace("?", "").replace("?", "")
                found.append(clean)
        return found

    def extract_interactions(self, text: str) -> list[str]:
        """提取相互作用药物"""
        found = []
        for drug in self.DRUG_INTERACTION_DRUGS:
            if drug in text:
                found.append(drug)
        return found


class KnowledgeGraph:
    """药物知识图谱"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.extractor = EntityExtractor()

    def build(self, documents: list[DrugDocument]):
        """从文档列表构建知识图谱"""
        for doc in documents:
            self._add_drug_node(doc)

            # 从各章节抽取关系
            for section, content in doc.sections.items():
                if section == "适应症":
                    self._extract_indications(doc, content)
                elif section == "不良反应":
                    self._extract_side_effects(doc, content)
                elif section == "禁忌":
                    self._extract_contraindications(doc, content)
                elif section == "药物相互作用":
                    self._extract_drug_interactions(doc, content)
                elif section == "用法用量":
                    self._extract_administration(doc, content)

        return self.graph

    def _add_drug_node(self, doc: DrugDocument):
        """添加药品节点"""
        self.graph.add_node(
            doc.drug_name,
            type="Drug",
            label=doc.drug_name[:20],
            file_path=doc.file_path,
        )

    def _extract_indications(self, doc: DrugDocument, text: str):
        """抽取适应症 → 疾病关系"""
        diseases = self.extractor.extract_diseases(text)
        populations = self.extractor.extract_populations(text)

        # 如果没有匹配到特定疾病，提取关键短语
        if not diseases:
            # 提取 "用于...治疗" 或 "适用于..." 的模式
            words = list(jieba.cut(text))
            # 用名词短语作为适应症标签
            diseases = [text[:30] + "..."]

        for disease in diseases:
            node_id = f"disease:{disease}"
            self.graph.add_node(node_id, type="Disease", label=disease)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="TREATS",
                label="治疗"
            )

        for pop in populations:
            node_id = f"population:{pop}"
            self.graph.add_node(node_id, type="Population", label=pop)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="APPLICABLE_TO",
                label="适用于"
            )

    def _extract_side_effects(self, doc: DrugDocument, text: str):
        """抽取不良反应 → 症状"""
        # 提取常见不良反应关键词
        side_effect_keywords = [
            "恶心", "呕吐", "腹泻", "头痛", "头晕", "皮疹", "瘙痒",
            "过敏", "发热", "寒颤", "血压下降", "心率减慢", "胸闷",
            "呼吸短促", "气喘", "红肿", "出血", "血小板聚集",
            "转氨酶异常", "血糖升高", "昼夜节律紊乱", "烧心",
            "腹部坠涨", "注射部位疼痛",
        ]
        for kw in side_effect_keywords:
            if kw in text:
                node_id = f"symptom:{kw}"
                self.graph.add_node(node_id, type="Symptom", label=kw)
                self.graph.add_edge(
                    doc.drug_name, node_id,
                    relation="HAS_SIDE_EFFECT",
                    label="不良反应"
                )

    def _extract_contraindications(self, doc: DrugDocument, text: str):
        """抽取禁忌"""
        # 提取禁忌条件
        diseases = self.extractor.extract_diseases(text)
        populations = self.extractor.extract_populations(text)

        for disease in diseases:
            node_id = f"disease:{disease}"
            if node_id not in self.graph:
                self.graph.add_node(node_id, type="Disease", label=disease)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="CONTRAINDICATED_FOR",
                label="禁忌"
            )

        for pop in populations:
            node_id = f"population:{pop}"
            if node_id not in self.graph:
                self.graph.add_node(node_id, type="Population", label=pop)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="CONTRAINDICATED_FOR",
                label="禁忌人群"
            )

        # 提取过敏禁忌
        if "过敏" in text:
            self.graph.add_edge(
                doc.drug_name, doc.drug_name,
                relation="CONTRAINDICATED_FOR",
                label="过敏者禁用"
            )

    def _extract_drug_interactions(self, doc: DrugDocument, text: str):
        """抽取药物相互作用"""
        interactions = self.extractor.extract_interactions(text)
        for drug in interactions:
            node_id = f"drug:{drug}"
            self.graph.add_node(node_id, type="Drug", label=drug)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="INTERACTS_WITH",
                label="相互作用"
            )

    def _extract_administration(self, doc: DrugDocument, text: str):
        """抽取给药方式"""
        routes = self.extractor.extract_routes(text)
        for route in routes:
            node_id = f"route:{route}"
            self.graph.add_node(node_id, type="Route", label=route)
            self.graph.add_edge(
                doc.drug_name, node_id,
                relation="ADMINISTERED_VIA",
                label="给药途径"
            )

    def query_drug(self, drug_name: str) -> dict:
        """查询某个药品的所有关系"""
        if drug_name not in self.graph:
            return {"error": f"未找到药品: {drug_name}"}

        result = {
            "drug": drug_name,
            "treats": [],
            "side_effects": [],
            "contraindications": [],
            "interactions": [],
            "routes": [],
        }

        for _, target, data in self.graph.out_edges(drug_name, data=True):
            rel = data.get("relation", "")
            target_label = self.graph.nodes[target].get("label", target)
            if rel == "TREATS":
                result["treats"].append(target_label)
            elif rel == "HAS_SIDE_EFFECT":
                result["side_effects"].append(target_label)
            elif rel == "CONTRAINDICATED_FOR":
                result["contraindications"].append(target_label)
            elif rel == "INTERACTS_WITH":
                result["interactions"].append(target_label)
            elif rel == "ADMINISTERED_VIA":
                result["routes"].append(target_label)

        return result

    def query_disease(self, disease: str) -> list[str]:
        """查询哪些药可以治疗某个疾病"""
        drugs = []
        disease_id = f"disease:{disease}"
        if disease_id in self.graph:
            for pred, _ in self.graph.in_edges(disease_id):
                if self.graph.nodes[pred].get("type") == "Drug":
                    drugs.append(pred)
        return drugs

    def search_entities(self, keyword: str) -> list[dict]:
        """模糊搜索实体"""
        results = []
        for node, data in self.graph.nodes(data=True):
            label = data.get("label", node)
            if keyword in label or keyword in node:
                results.append({
                    "id": node,
                    "type": data.get("type", "Unknown"),
                    "label": label,
                    "degree": self.graph.degree(node),
                })
        return sorted(results, key=lambda x: x["degree"], reverse=True)

    def get_subgraph(self, drug_name: str, depth: int = 1) -> nx.DiGraph:
        """获取某个药品为中心的子图"""
        if drug_name not in self.graph:
            return None
        nodes = {drug_name}
        current = {drug_name}
        for _ in range(depth):
            neighbors = set()
            for n in current:
                neighbors.update(self.graph.successors(n))
                neighbors.update(self.graph.predecessors(n))
            nodes.update(neighbors)
            current = neighbors
        return self.graph.subgraph(nodes)

    def export_json(self, output_path: str):
        """导出图谱为 JSON (节点+边)"""
        data = {
            "nodes": [],
            "edges": [],
        }
        for node, attrs in self.graph.nodes(data=True):
            data["nodes"].append({
                "id": node,
                "type": attrs.get("type", "Unknown"),
                "label": attrs.get("label", node),
            })
        for src, dst, attrs in self.graph.edges(data=True):
            data["edges"].append({
                "source": src,
                "target": dst,
                "relation": attrs.get("relation", ""),
                "label": attrs.get("label", ""),
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    def export_html(self, output_path: str = "./data/knowledge_graph.html"):
        """导出为 PyVis 交互式 HTML"""
        try:
            from pyvis.network import Network
        except ImportError:
            print("请安装 pyvis: pip install pyvis")
            return None

        net = Network(height="750px", width="100%", directed=True, notebook=False)

        color_map = {
            "Drug": "#4CAF50",
            "Disease": "#FF9800",
            "Symptom": "#F44336",
            "Population": "#2196F3",
            "Route": "#9C27B0",
        }

        for node, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "Unknown")
            color = color_map.get(ntype, "#757575")
            net.add_node(
                node,
                label=attrs.get("label", node),
                color=color,
                title=f"Type: {ntype}",
            )

        edge_colors = {
            "TREATS": "#4CAF50",
            "HAS_SIDE_EFFECT": "#F44336",
            "CONTRAINDICATED_FOR": "#FF5722",
            "INTERACTS_WITH": "#FFC107",
            "ADMINISTERED_VIA": "#2196F3",
            "APPLICABLE_TO": "#9C27B0",
        }

        for src, dst, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "")
            color = edge_colors.get(rel, "#757575")
            net.add_edge(
                src, dst,
                label=attrs.get("label", ""),
                color=color,
                arrows="to",
            )

        net.set_options("""
        {
          "nodes": {"font": {"size": 14, "face": "Microsoft YaHei"}},
          "edges": {"font": {"size": 10, "face": "Microsoft YaHei"}, "smooth": {"type": "curvedCW"}},
          "physics": {"barnesHut": {"gravitationalConstant": -3000, "springLength": 200}}
        }
        """)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        return output_path

    @property
    def summary(self) -> dict:
        """图谱统计摘要"""
        types = defaultdict(int)
        for _, attrs in self.graph.nodes(data=True):
            types[attrs.get("type", "Unknown")] += 1

        rels = defaultdict(int)
        for _, _, attrs in self.graph.edges(data=True):
            rels[attrs.get("relation", "Unknown")] += 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(types),
            "relation_types": dict(rels),
        }


if __name__ == "__main__":
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

    print("Step 1: 解析文档...")
    parser = DrugMDParser(md_dir)
    docs = parser.parse_all()
    print(f"  文档数: {len(docs)}")

    print("\nStep 2: 构建知识图谱...")
    kg = KnowledgeGraph()
    kg.build(docs)
    summary = kg.summary
    print(f"  节点数: {summary['nodes']}, 边数: {summary['edges']}")
    print(f"  节点类型: {summary['node_types']}")
    print(f"  关系类型: {summary['relation_types']}")

    print("\nStep 3: 查询示例...")
    # 查询单个药品
    result = kg.query_drug("丙泊酚中／长链脂肪乳注射液")
    print(f"\n  丙泊酚 知识图谱:")
    for k, v in result.items():
        if k != "drug" and v:
            print(f"    {k}: {', '.join(v)}")

    # 查询疾病可用的药
    drugs = kg.query_disease("糖尿病")
    print(f"\n  治疗'糖尿病'的药品: {drugs}")

    # 搜索实体
    results = kg.search_entities("肝")
    print(f"\n  包含'肝'的实体:")
    for r in results:
        print(f"    [{r['type']}] {r['label']} (度={r['degree']})")

    # 导出
    print("\nStep 4: 导出...")
    json_path = kg.export_json("./data/knowledge_graph.json")
    print(f"  JSON: {json_path}")
    html_path = kg.export_html("./data/knowledge_graph.html")
    print(f"  HTML: {html_path}")
