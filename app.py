import os
import pickle
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 从 .env 文件加载配置
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

BM25_CACHE_PATH = "./bm25_cache.pkl"


# ==========================================
# 2. 系统初始化与缓存
# ==========================================
@st.cache_resource
def init_system():
    # 加载向量模型和数据库
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 构建 BM25 检索器（优先从磁盘缓存加载，避免每次冷启动重建）
    if os.path.exists(BM25_CACHE_PATH):
        with open(BM25_CACHE_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        db_data = vectordb.get()
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(db_data['documents'], db_data['metadatas'])]
        bm25_retriever = BM25Retriever.from_documents(docs)
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)

    # 加载重排模型
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

    return vectordb, bm25_retriever, reranker


# ==========================================
# 3. 翻译缓存（避免重复调用 LLM 翻译相同 query）
# ==========================================
def get_translation_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}

def translate_query(llm, query: str) -> str:
    cache_key = get_translation_cache_key(query)
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
    translate_prompt = (
        f"将以下中文 CAD 提问翻译为精确的英文，保留原意，直接输出英文无需解释：{query}"
    )
    result = llm.invoke(translate_prompt).content
    st.session_state.translation_cache[cache_key] = result
    return result


# ==========================================
# 4. RAG 检索逻辑
# ==========================================
def perform_rag_retrieval(query, vectordb, bm25_retriever, reranker, rrf_k=60):
    # 双路召回
    dense_docs = vectordb.similarity_search(query, k=50)
    bm25_retriever.k = 50
    sparse_docs = bm25_retriever.invoke(query)

    # RRF 融合
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

    fused_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:50]
    top_50_docs = [item["doc"] for item in fused_docs]

    # Cross-Encoder 重排
    pairs = [[query, doc.page_content] for doc in top_50_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(top_50_docs, scores):
        doc.metadata['rerank_score'] = score
    top_50_docs.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)

    return top_50_docs[:5]


# ==========================================
# 5. 前端 UI 与大模型交互逻辑
# ==========================================
st.set_page_config(page_title="OCCT/AS 专属知识大脑", page_icon="🧠", layout="wide")
st.title("⚙️ CAD 内核开发辅助问答系统 (RAG)")

vectordb, bm25_retriever, reranker = init_system()

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.1,
    streaming=True
)

prompt_template = ChatPromptTemplate.from_template("""
你是一个精通 C++ 和三维几何内核（如 OpenCascade, Analysis Situs）的资深开发专家。
请【仅根据】以下提供的上下文信息，客观且准确地回答用户的问题。

严格要求：
1. 如果上下文中没有提及能够回答该问题的信息，请直接回答"未在文档库中找到相关信息"，绝对不要根据自身的知识进行编造（即严禁幻觉）。
2. 在回答时，如果引用了具体的 C++ 类名或函数，请保持原有格式（例如：`BRepBuilderAPI_MakeSolid`）。

=== 上下文信息 ===
{context}
=== === === ===

用户提问: {question}
专家解答:
""")

chain = prompt_template | llm | StrOutputParser()

# 对话状态管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入处理
if user_input := st.chat_input("询问关于 OCCT, AAG, B-Rep 拓扑操作的任何问题..."):
    # 存储用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 翻译 query（带缓存）
    english_query = translate_query(llm, user_input)

    with st.chat_message("assistant"):
        with st.status(f"正在检索: {english_query}...", expanded=True) as status:
            final_docs = perform_rag_retrieval(english_query, vectordb, bm25_retriever, reranker)
            context_str = "\n\n".join(
                [f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}" for d in final_docs]
            )
            st.write("检索完成！获取到以下核心上下文：")
            for d in final_docs:
                st.info(
                    f"得分: {d.metadata.get('rerank_score', 0):.2f} | "
                    f"来源: {os.path.basename(d.metadata.get('source', '未知'))}"
                )
            status.update(label="文档阅读完毕，正在生成解答...", state="complete", expanded=False)

        # 流式输出
        response_placeholder = st.empty()
        full_response = ""
        for chunk in chain.stream({"context": context_str, "question": user_input}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    # 存储 assistant 消息
    st.session_state.messages.append({"role": "assistant", "content": full_response})
