# CAD 内核开发辅助问答系统 (RAG)

基于 OpenCascade 和 Analysis Situs 文档构建的本地知识库问答系统，采用混合检索增强生成（RAG）架构。

## 系统架构

```
用户提问 (中文)
    │
    ▼
Query Translation (LLM 前置翻译为英文)
    │
    ├──────────────────────┐
    ▼                      ▼
向量检索 (Dense)        BM25 检索 (Sparse)
BAAI/bge-small-en-v1.5   关键词匹配
Top-50                   Top-50
    │                      │
    └──────────┬───────────┘
               ▼
         RRF 融合排序
    (Reciprocal Rank Fusion)
               │
               ▼
    Cross-Encoder 重排
  ms-marco-MiniLM-L-6-v2
               │
               ▼
           Top-5 上下文
               │
               ▼
        LLM 生成最终回答
       (DeepSeek / 兼容 OpenAI 格式)
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `step1_build_db.py` | 扫描 `data/raw_docs` 下的 Markdown 文档，切分并写入 ChromaDB |
| `step2_hybrid_retrieval.py` | 混合检索管道的命令行调试脚本，用于在接入前端前独立验证"双路召回→RRF→重排"流程是否正常，输入硬编码问题并打印 Top 5 结果 |
| `step3_evaluation.py` | 自动生成测试集并执行三路对比评估 |
| `app.py` | Streamlit 前端问答界面 |
| `.env` | API 密钥配置（不提交到 git） |

## 快速开始

**1. 安装依赖**

```bash
pip install streamlit langchain-chroma langchain-huggingface langchain-openai \
            langchain-community sentence-transformers python-dotenv
```

**2. 配置 API**

复制 `.env.example` 为 `.env` 并填入你的 API 信息：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL_NAME=deepseek-chat
```

**3. 构建知识库**

将 Markdown 文档放入 `data/raw_docs/`，然后运行：

```bash
python step1_build_db.py
```

**4. 启动问答界面**

```bash
streamlit run app.py
```

**5. 运行评估**

```bash
python step3_evaluation.py
```

## 评估结果

测试集：从 ChromaDB 随机抽取 20 个 chunk，由 LLM 以开发者真实提问风格反向生成对应问题，构成 20 条 (Query, Ground Truth) 对。指标为 Hit Rate @ 5。

| 检索方式 | 命中数 | Hit Rate @ 5 |
|----------|--------|--------------|
| 纯向量检索 (Dense-only) | 14/20 | 70.0% |
| 纯 BM25 检索 (Sparse-only) | 10/20 | 50.0% |
| 混合检索 + Cross-Encoder 重排 | 17/20 | **85.0%** |

混合检索相比纯向量提升 **+15%**，相比纯 BM25 提升 **+35%**。


## 技术栈

- **向量数据库**: ChromaDB
- **Embedding 模型**: BAAI/bge-small-en-v1.5
- **重排模型**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: DeepSeek Chat（兼容 OpenAI API 格式）
- **框架**: LangChain + Streamlit
