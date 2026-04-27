from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

CHROMA_DB_DIR = "./chroma_db"


def load_vector_db():
    print("正在加载底层的 Chroma 向量数据库...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    return vectordb


def build_bm25_retriever(vectordb):
    print("正在从向量库中提取数据，构建 BM25 稀疏检索引擎...")
    db_data = vectordb.get()
    docs = [Document(page_content=t, metadata=m)
            for t, m in zip(db_data['documents'], db_data['metadatas'])]
    return BM25Retriever.from_documents(docs)


def rrf_fusion(dense_docs, sparse_docs, rrf_k=60):
    """
    倒数排名融合 (Reciprocal Rank Fusion)
    rrf_k: RRF 公式中的平滑常数，与返回数量无关，默认 60
    """
    print("正在执行倒数排名融合 (RRF)...")
    rrf_scores = {}
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (rrf_k + rank + 1)
    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (rrf_k + rank + 1)
    fused_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in fused_results]


def main():
    query = "How to construct a B-Rep solid from faces in OpenCascade?"
    print(f"\n>>> 原始提问: '{query}'\n")

    vectordb = load_vector_db()
    bm25_retriever = build_bm25_retriever(vectordb)

    # 阶段一：双路召回 (各取 Top 50)
    print("阶段一：执行双路召回 (Top 50)...")
    dense_results = vectordb.as_retriever(search_kwargs={"k": 50}).invoke(query)
    bm25_retriever.k = 50
    sparse_results = bm25_retriever.invoke(query)

    # 阶段二：RRF 融合，取前 50 进入重排
    fused_docs = rrf_fusion(dense_results, sparse_results)[:50]

    # 阶段三：Cross-Encoder 重排
    print("阶段三：启动 Cross-Encoder 进行语义重排...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    pairs = [[query, doc.page_content] for doc in fused_docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(fused_docs, scores):
        doc.metadata['rerank_score'] = score
    fused_docs.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)

    final_top_5 = fused_docs[:5]

    print("\n" + "=" * 50)
    print(">>> 混合检索与重排完成！最终入选的 Top 5 片段：")
    print("=" * 50 + "\n")

    for i, doc in enumerate(final_top_5):
        source = doc.metadata.get('source', 'Unknown')
        score = doc.metadata.get('rerank_score', 0.0)
        print(f"[{i + 1}] 来源: {source} (重排得分: {score:.4f})")
        print(f"片段摘录: {doc.page_content[:200]}...\n")


if __name__ == "__main__":
    main()
