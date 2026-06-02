#!/usr/bin/env python3
"""
一键构建：解析文档 → 向量库 → 知识图谱 → 导出
Usage: python build_all.py [MD目录路径]
"""
import sys
from pathlib import Path
from parse_docs import DrugMDParser
from build_vector_store import VectorStore
from build_knowledge_graph import KnowledgeGraph


def main():
    md_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

    print("╔══════════════════════════════════════════╗")
    print("║  药物说明书 RAG 系统 — 一键构建            ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n📂 数据目录: {md_dir}")

    # Step 1: 解析
    print("\n" + "─" * 44)
    print("  [1/3] 解析 Markdown 文档")
    parser = DrugMDParser(md_dir)
    docs = parser.parse_all()
    chunks = parser.get_all_chunks()
    print(f"  ✓ {len(docs)} 份说明书 → {len(chunks)} 个切片")

    # Step 2: 向量库
    print("\n" + "─" * 44)
    print("  [2/3] 构建向量知识库 (ChromaDB + BGE)")
    vs = VectorStore(persist_dir="./data/vector_store")
    vs.build(chunks)
    print(f"  ✓ {vs.get_collection().count()} 条向量记录")

    # Step 3: 知识图谱
    print("\n" + "─" * 44)
    print("  [3/3] 构建知识图谱 (NetworkX)")
    kg = KnowledgeGraph()
    kg.build(docs)
    s = kg.summary
    print(f"  ✓ 节点: {s['nodes']} | 边: {s['edges']}")
    print(f"  ✓ 类型: {s['node_types']}")
    print(f"  ✓ 关系: {s['relation_types']}")

    # 导出
    json_path = kg.export_json("./data/knowledge_graph.json")
    html_path = kg.export_html("./data/knowledge_graph.html")

    print("\n" + "═" * 44)
    print("  ✅ 构建完成！")
    print(f"  📊 向量库: ./data/vector_store/")
    print(f"  🧠 图谱JSON: {json_path}")
    print(f"  🌐 图谱HTML: {html_path}")
    print("═" * 44)

    # 快速测试
    print("\n🧪 快速验证:")
    test_q = "糖尿病"
    drugs = kg.query_disease(test_q)
    print(f"  查询'{test_q}' → 药品: {drugs}")

    vr = vs.search("麻醉诱导剂量", n_results=1)
    if vr:
        print(f"  向量搜索'麻醉诱导剂量' → {vr[0]['drug_name']} (score={vr[0]['score']:.3f})")


if __name__ == "__main__":
    main()
