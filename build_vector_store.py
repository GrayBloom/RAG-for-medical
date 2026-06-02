"""
向量知识库构建模块
使用 ChromaDB + sentence-transformers (中文 embedding)
"""
import os
import json
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from parse_docs import DrugMDParser, DrugChunk


class VectorStore:
    """基于 ChromaDB 的向量知识库"""

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"  # fallback

    @staticmethod
    def _resolve_model_path() -> str:
        """尝试从本地 modelscope 缓存加载模型，失败则回退 HF"""
        import os
        # 优先级：bge-small-zh (更好的检索模型) > corom (fallback)
        candidates = [
            "./data/models/AI-ModelScope/bge-small-zh-v1.5",
            "./data/models/iic/nlp_corom_sentence-embedding_chinese-base",
        ]
        for local in candidates:
            if os.path.isdir(local):
                return os.path.abspath(local)
        return VectorStore.DEFAULT_MODEL

    def __init__(
        self,
        persist_dir: str = "./data/vector_store",
        model_name: str = DEFAULT_MODEL,
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 初始化 embedding 模型
        model_path = self._resolve_model_path()
        print(f"加载 embedding 模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def build(self, chunks: list[DrugChunk], collection_name: str = "drug_docs"):
        """
        从文档切片构建向量库
        先删除旧集合再重建（幂等）
        """
        # 删除已有集合
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 批量编码和入库
        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "drug_name": c.drug_name,
                "section": c.section,
                "type": c.metadata.get("type", "unknown"),
            }
            for c in chunks
        ]

        print(f"正在编码 {len(texts)} 个文本片段...")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()

        # 分批写入（ChromaDB 有批量大小限制）
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"向量库构建完成: {collection.count()} 条记录")
        return collection

    def get_collection(self, name: str = "drug_docs"):
        """获取已有集合"""
        return self.client.get_collection(name)

    def search(
        self,
        query: str,
        collection_name: str = "drug_docs",
        n_results: int = 5,
        drug_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        语义搜索
        Args:
            query: 查询文本
            n_results: 返回结果数
            drug_filter: 可选，仅搜索特定药品
        """
        collection = self.get_collection(collection_name)
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        where_filter = None
        if drug_filter:
            where_filter = {"drug_name": drug_filter}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 整理结果
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "rank": i + 1,
                "id": results["ids"][0][i],
                "drug_name": results["metadatas"][0][i]["drug_name"],
                "section": results["metadatas"][0][i]["section"],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            })
        return formatted

    def search_by_section(self, section: str, query: str, n_results: int = 5):
        """在特定章节范围内搜索"""
        collection = self.get_collection("drug_docs")
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"section": section},
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "rank": i + 1,
                "drug_name": results["metadatas"][0][i]["drug_name"],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return formatted


if __name__ == "__main__":
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Python_Program/RAG/cleaned_MD_optimized"

    # 解析文档
    print("=" * 50)
    print("Step 1: 解析文档")
    parser = DrugMDParser(md_dir)
    parser.parse_all()
    chunks = parser.get_all_chunks()
    print(f"  文档数: {len(parser.documents)}, 切片数: {len(chunks)}")

    # 构建向量库
    print("\n" + "=" * 50)
    print("Step 2: 构建向量知识库")
    vs = VectorStore(persist_dir="./data/vector_store")
    vs.build(chunks)

    # 测试搜索
    print("\n" + "=" * 50)
    print("Step 3: 测试搜索")
    test_queries = [
        "糖尿病用什么药？",
        "麻醉诱导的剂量是多少？",
        "肝功能不全患者需要注意什么？",
    ]
    for q in test_queries:
        print(f"\n🔍 查询: {q}")
        results = vs.search(q, n_results=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['drug_name']} | {r['section']}")
            print(f"    {r['content'][:100]}...")
