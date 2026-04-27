"""
step3_evaluation.py — RAG 系统自动化评估脚本

评估指标说明：
  Hit Rate @ 5：
    对于每个测试问题，将其送入检索管道，取 Top 5 结果。
    若 Top 5 中任意一个 chunk 的 page_content 包含（或高度相似于）
    生成该问题时所用的原始 Ground Truth 文本，则视为"命中"。
    Hit Rate @ 5 = 命中数 / 总问题数
    该指标衡量的是检索系统的召回能力，而非生成质量。
"""

import os
import json
import time
import random
import pickle
import difflib

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

# ==========================================
# 配置
# ==========================================
load_dotenv()
API_KEY       = os.getenv("OPENAI_API_KEY")
BASE_URL      = os.getenv("OPENAI_BASE_URL")
MODEL_NAME    = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

CHROMA_DB_DIR   = "./chroma_db"
BM25_CACHE_PATH = "./bm25_cache.pkl"
EVAL_DATASET    = "./eval_dataset.json"

SAMPLE_SIZE     = 20   # 抽取的 chunk 数量
MIN_CHUNK_LEN   = 200  # 过滤掉内容太短的 chunk（字符数）
TOP_K_RETRIEVE  = 50   # 双路召回各取 Top K
RRF_K           = 60   # RRF 平滑常数
TOP_N_RERANK    = 5    # 重排后取 Top N
SIMILARITY_THR  = 0.5  # 相似度判断阈值（SequenceMatcher ratio）
MAX_RETRIES     = 3    # LLM 调用最大重试次数


# ==========================================
# 工具函数
# ==========================================
def print_section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def print_step(idx: int, total: int, msg: str):
    print(f"  [{idx:>2}/{total}] {msg}")


def is_similar(text_a: str, text_b: str, threshold: float = SIMILARITY_THR) -> bool:
    """
    用 SequenceMatcher 计算两段文本的相似度。
    先做字符串包含判断（快速路径），再做模糊匹配（兜底）。
    """
    # 快速路径：直接包含
    if text_b[:150] in text_a:
        return True
    # 模糊匹配：取前 300 字符比较，避免超长文本拖慢速度
    ratio = difflib.SequenceMatcher(None, text_a[:500], text_b[:300]).ratio()
    return ratio >= threshold


# ==========================================
# 1. 初始化模型与数据库
# ==========================================
def init_resources():
    print_section("初始化：加载模型与数据库")

    print("  ▸ 加载 Embedding 模型 (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("  ▸ 连接 ChromaDB...")
    vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    print("  ▸ 构建 BM25 检索器（优先读取磁盘缓存）...")
    if os.path.exists(BM25_CACHE_PATH):
        with open(BM25_CACHE_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        print("    ✔ 从缓存加载 BM25 完成")
    else:
        db_data = vectordb.get()
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(db_data['documents'], db_data['metadatas'])]
        bm25_retriever = BM25Retriever.from_documents(docs)
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("    ✔ BM25 构建完成并已缓存")

    print("  ▸ 加载 Cross-Encoder 重排模型...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

    print("  ▸ 初始化 LLM...")
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0.2,
    )

    print("  ✔ 所有资源初始化完成\n")
    return vectordb, bm25_retriever, reranker, llm


# ==========================================
# 2. 任务一：生成测试集
# ==========================================
def generate_question_with_retry(llm, chunk_text: str) -> str | None:
    """调用 LLM 为给定 chunk 生成一个英文技术问题，失败时自动重试。"""
    prompt = (
        "You are a technical question generator for a CAD/geometry kernel knowledge base.\n"
        "Based ONLY on the following document excerpt, generate exactly ONE specific technical "
        "question in English that can be answered using this excerpt.\n"
        "Output only the question itself, no explanation, no numbering.\n\n"
        f"Document excerpt:\n{chunk_text[:600]}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = llm.invoke(prompt).content.strip()
            if result.endswith("?") or len(result) > 10:
                return result
        except Exception as e:
            wait = 2 ** attempt  # 指数退避：2s, 4s, 8s
            print(f"      ⚠ LLM 调用失败 (第 {attempt} 次): {e}，{wait}s 后重试...")
            time.sleep(wait)
    return None


def generate_eval_dataset(vectordb, llm) -> list[dict]:
    print_section("任务一：自动生成测试集")

    # 从 ChromaDB 拉取全量数据，过滤短 chunk，随机抽样
    print(f"  ▸ 从 ChromaDB 抽取 {SAMPLE_SIZE} 个有效 chunk...")
    db_data = vectordb.get()
    all_docs = [
        {"content": t, "source": m.get("source", "unknown")}
        for t, m in zip(db_data['documents'], db_data['metadatas'])
        if len(t) >= MIN_CHUNK_LEN
    ]
    print(f"    有效 chunk 总数（长度 ≥ {MIN_CHUNK_LEN} 字符）: {len(all_docs)}")

    sampled = random.sample(all_docs, min(SAMPLE_SIZE, len(all_docs)))

    dataset = []
    total = len(sampled)
    for i, item in enumerate(sampled, 1):
        print_step(i, total, f"生成问题中... 来源: {os.path.basename(item['source'])}")
        question = generate_question_with_retry(llm, item["content"])
        if question:
            dataset.append({
                "question": question,
                "ground_truth": item["content"],
                "source": item["source"]
            })
            print(f"      ✔ Q: {question[:80]}{'...' if len(question) > 80 else ''}")
        else:
            print(f"      ✘ 跳过该 chunk（LLM 多次失败）")

    # 保存到本地
    with open(EVAL_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n  ✔ 测试集已保存至 {EVAL_DATASET}（共 {len(dataset)} 条）")
    return dataset


# ==========================================
# 3. 三种检索管道
# ==========================================
def dense_retrieve(query: str, vectordb) -> list[Document]:
    """纯向量检索，直接取 Top N。"""
    return vectordb.similarity_search(query, k=TOP_N_RERANK)


def bm25_retrieve(query: str, bm25_retriever) -> list[Document]:
    """纯 BM25 稀疏检索，取 Top N。"""
    bm25_retriever.k = TOP_N_RERANK
    return bm25_retriever.invoke(query)


def hybrid_retrieve(query: str, vectordb, bm25_retriever, reranker) -> list[Document]:
    """双路召回 -> RRF 融合 -> Cross-Encoder 重排，返回 Top N 结果。"""
    dense_docs = vectordb.similarity_search(query, k=TOP_K_RETRIEVE)
    bm25_retriever.k = TOP_K_RETRIEVE
    sparse_docs = bm25_retriever.invoke(query)

    rrf_scores: dict[str, dict] = {}
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)
    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    top_docs = [item["doc"] for item in fused[:TOP_K_RETRIEVE]]

    pairs = [[query, doc.page_content] for doc in top_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(top_docs, scores):
        doc.metadata["rerank_score"] = float(score)
    top_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

    return top_docs[:TOP_N_RERANK]


def eval_pipeline(name: str, dataset: list[dict], retrieve_fn) -> tuple[int, int, float]:
    """通用评估循环，返回 (hits, total, hit_rate)。"""
    hits = 0
    total = len(dataset)
    print(f"\n  ── {name} ──")
    for i, item in enumerate(dataset, 1):
        retrieved = retrieve_fn(item["question"])
        hit = any(is_similar(doc.page_content, item["ground_truth"]) for doc in retrieved)
        hits += hit
        status = "✔" if hit else "✘"
        print_step(i, total, f"{status} {item['question'][:65]}{'...' if len(item['question']) > 65 else ''}")
    hit_rate = hits / total if total > 0 else 0.0
    print(f"     → 命中 {hits}/{total}，Hit Rate @ {TOP_N_RERANK} = {hit_rate:.1%}")
    return hits, total, hit_rate


# ==========================================
# 4. 任务二：三路对比评估
# ==========================================
def run_evaluation(dataset: list[dict], vectordb, bm25_retriever, reranker):
    print_section("任务二：三路对比评估（Dense / BM25 / Hybrid+Rerank）")

    hits_dense, total, rate_dense = eval_pipeline(
        "① 纯向量检索 (Dense-only)",
        dataset,
        lambda q: dense_retrieve(q, vectordb)
    )

    hits_bm25, _, rate_bm25 = eval_pipeline(
        "② 纯 BM25 检索 (Sparse-only)",
        dataset,
        lambda q: bm25_retrieve(q, bm25_retriever)
    )

    hits_hybrid, _, rate_hybrid = eval_pipeline(
        "③ 混合检索 + Cross-Encoder 重排 (Hybrid+Rerank)",
        dataset,
        lambda q: hybrid_retrieve(q, vectordb, bm25_retriever, reranker)
    )

    # ==========================================
    # 5. 打印对比报告
    # ==========================================
    print_section("最终评估报告")
    print(f"  配置: Top-K召回={TOP_K_RETRIEVE}, RRF_K={RRF_K}, Top-N={TOP_N_RERANK}, 相似度阈值={SIMILARITY_THR}")
    print(f"  测试集大小: {total} 条\n")
    print(f"  {'检索方式':<30} {'命中数':>6}  {'Hit Rate @ 5':>12}")
    print(f"  {'-'*52}")
    print(f"  {'① 纯向量检索 (Dense-only)':<30} {hits_dense:>4}/{total}  {rate_dense:>11.1%}")
    print(f"  {'② 纯BM25检索 (Sparse-only)':<30} {hits_bm25:>4}/{total}  {rate_bm25:>11.1%}")
    print(f"  {'③ 混合检索+重排 (Hybrid+Rerank)':<30} {hits_hybrid:>4}/{total}  {rate_hybrid:>11.1%}")
    print(f"  {'-'*52}")

    best = max(rate_dense, rate_bm25, rate_hybrid)
    gain_vs_dense = rate_hybrid - rate_dense
    gain_vs_bm25  = rate_hybrid - rate_bm25
    print(f"\n  混合检索相比纯向量提升: {gain_vs_dense:+.1%}")
    print(f"  混合检索相比纯BM25提升: {gain_vs_bm25:+.1%}")
    print()


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    vectordb, bm25_retriever, reranker, llm = init_resources()

    # 任务一：若测试集已存在则直接加载，否则重新生成
    if os.path.exists(EVAL_DATASET):
        print(f"\n  ℹ 检测到已有测试集 {EVAL_DATASET}，直接加载。")
        print(f"    如需重新生成，请删除该文件后重新运行。")
        with open(EVAL_DATASET, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"    已加载 {len(dataset)} 条测试数据。")
    else:
        dataset = generate_eval_dataset(vectordb, llm)

    # 任务二：执行评估
    run_evaluation(dataset, vectordb, bm25_retriever, reranker)
