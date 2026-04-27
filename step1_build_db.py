from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 路径配置
DATA_PATH = "./data/raw_docs"
CHROMA_DB_DIR = "./chroma_db"


def load_documents(data_path):
    print(f"正在扫描 {data_path} 下的 Markdown 文档...")
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"成功加载了 {len(docs)} 篇原始文档。")
    return docs


def split_documents(docs):
    print("开始按逻辑切分文档 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"切分完成：转化为 {len(chunks)} 个文本块 (Chunks)。")
    return chunks


def init_vector_db():
    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"连接本地 Chroma 数据库: {CHROMA_DB_DIR}")
    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    return vectordb


def add_to_db(vectordb, chunks):
    print("正在计算向量并存入数据库 (这可能需要几分钟，取决于文档数量)...")
    vectordb.add_documents(chunks)
    # 新版 Chroma 自动持久化，无需手动调用 persist()
    print(f"入库完成！当前数据库中共有 {vectordb._collection.count()} 条记录。")


if __name__ == "__main__":
    raw_documents = load_documents(DATA_PATH)
    if len(raw_documents) > 0:
        document_chunks = split_documents(raw_documents)
        db = init_vector_db()
        add_to_db(db, document_chunks)
    else:
        print("未找到文档，请检查 data/raw_docs 目录下是否有 .md 文件。")
