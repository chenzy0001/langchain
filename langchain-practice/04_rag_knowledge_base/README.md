# 04 RAG 知识库与检索增强生成

## 一、理论教学

### 4.1 什么是 RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是解决 LLM 知识局限性的核心方案：

```
传统 LLM：问题 → LLM（训练时的知识）→ 答案（可能过时/幻觉）

RAG：    问题 → 检索器（实时知识库）→ 相关文档 → LLM + 文档 → 准确答案
```

**为什么需要 RAG？**
- LLM 训练数据有截止日期，无法处理最新信息
- LLM 无法访问企业内部私有文档
- LLM 在垂直领域（医疗/法律/金融）容易产生幻觉
- RAG 的知识更新成本远低于微调（Fine-tuning）

### 4.2 RAG Pipeline 全流程

```
【离线阶段：知识库构建】
原始文档（PDF/网页/数据库）
    ↓ DocumentLoader
文档对象（Document）
    ↓ TextSplitter（分块）
文本块（chunks）
    ↓ Embedding Model（向量化）
向量 + 元数据
    ↓ VectorStore（存储）
向量数据库（FAISS/Chroma/Pinecone）

【在线阶段：查询检索】
用户问题
    ↓ Embedding Model（向量化）
查询向量
    ↓ 相似度搜索（cosine/dot product）
Top-K 相关文档块
    ↓ 上下文构建
Prompt = 系统指令 + 检索结果 + 用户问题
    ↓ LLM
最终答案
```

### 4.3 文档分块策略（Chunking）

分块是 RAG 质量的关键，不同策略适用不同场景：

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

# 策略1：递归字符分割（最常用，自动适应文本结构）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # 每块目标字符数
    chunk_overlap=200,       # 块间重叠（保持上下文连续性）
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 优先按段落分割
    length_function=len,
)

# 策略2：按 Markdown 标题分割（结构化文档）
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "标题1"),
        ("##", "标题2"),
        ("###", "标题3"),
    ]
)

# 策略3：按 Token 分割（精确控制 Token 预算）
token_splitter = TokenTextSplitter(
    chunk_size=256,    # 每块 256 tokens
    chunk_overlap=50,
)
```

**分块参数选择指南：**
| 文档类型 | chunk_size | chunk_overlap | 推荐分割器 |
|---------|-----------|--------------|-----------|
| 技术文档 | 500-1000 | 100-200 | RecursiveCharacter |
| Markdown | 按标题 | 50-100 | MarkdownHeader |
| 代码 | 按函数 | 0-50 | Language |
| 新闻/文章 | 300-500 | 50-100 | RecursiveCharacter |
| 长合同/法律 | 1000-2000 | 200-400 | Token |

### 4.4 向量检索优化

**基础相似度搜索：**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="knowledge_base",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# 方法1：相似度搜索
docs = vectorstore.similarity_search(query, k=4)

# 方法2：带分数的相似度搜索
docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=4)
# 过滤低相关性文档
relevant_docs = [(doc, score) for doc, score in docs_with_scores if score > 0.7]

# 方法3：MMR（Maximal Marginal Relevance）减少冗余
docs_mmr = vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=20, lambda_mult=0.5  # lambda_mult: 0=多样性优先, 1=相关性优先
)
```

**高级检索：Ensemble Retriever（混合检索）**

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever

# BM25：关键词匹配（精确词语）
bm25_retriever = BM25Retriever.from_documents(docs, k=4)

# FAISS：语义检索（语义相似）
from langchain_community.vectorstores import FAISS
faiss_vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

# 混合检索：结合两种策略（BM25 权重 0.5 + 语义 权重 0.5）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
)
```

### 4.5 RAG Chain 构建

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def format_docs(docs: list) -> str:
    """将检索到的文档格式化为 Prompt 上下文。"""
    return "\n\n".join(
        f"[来源{i+1}: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的知识问答助手。
    
请基于以下检索到的上下文回答问题。
- 如果上下文中包含相关信息，优先使用上下文内容
- 如果上下文不包含答案，说明"根据现有知识库无法回答"
- 在回答末尾注明使用了哪些来源

上下文：
{context}"""),
    ("human", "{question}"),
])

def build_rag_chain(retriever):
    """构建标准 RAG Chain。"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 并行执行：检索 + 传递问题
    rag_chain_with_source = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    ) | RAG_PROMPT | llm | StrOutputParser()
    
    return rag_chain_with_source
```

---

## 二、实践练习

### 题目 1：构建带来源引用的 RAG 问答系统

**背景：** 企业内部知识库问答，需要明确标注答案来源，防止"幻觉"。

**要求：**
1. 使用 `RecursiveCharacterTextSplitter` 对模拟文档进行分块
2. 使用 `CacheBackedEmbeddings` 缓存向量计算结果
3. 构建 RAG Chain，答案必须包含来源引用（文档名 + 段落编号）
4. 实现 `answer_with_sources()` 函数，同时返回答案和引用来源列表

---

### 题目 1 标准答案

```python
"""
题目1：带来源引用的 RAG 问答系统
知识点：文档分块、Embedding 缓存、来源追踪、RAG Chain
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os


# ===== 模拟企业知识库文档 =====
SAMPLE_DOCS = [
    Document(
        page_content="""LangChain 是一个开源框架，用于构建基于大型语言模型（LLM）的应用程序。
它提供了丰富的组件，包括 Prompt 模板、LLM 封装、Chain、Agent、Memory 和 Tool。
LangChain 的核心设计理念是"可组合性"——通过链式组合各个组件来构建复杂的 AI 应用。
LangChain Expression Language（LCEL）是 LangChain 2.0 引入的声明式编程接口，
使用 | 操作符连接各个组件，支持流式输出、批处理和异步执行。""",
        metadata={"source": "langchain_intro.txt", "section": "概述"},
    ),
    Document(
        page_content="""RAG（检索增强生成）是 LangChain 最常见的应用场景之一。
基本流程包括：文档加载、文档分块、向量化、向量存储、语义检索、上下文注入和答案生成。
LangChain 支持多种向量数据库，包括 FAISS（本地）、Chroma（本地/远程）、
Pinecone（云端）、Weaviate、Qdrant 等。
选择向量数据库时，需要考虑数据量、检索速度、部署方式和成本。""",
        metadata={"source": "rag_guide.txt", "section": "RAG基础"},
    ),
    Document(
        page_content="""Agent 是 LangChain 中最灵活的组件，允许 LLM 动态决定调用哪些工具。
主要的 Agent 类型包括：
1. ReAct Agent：通过 Thought-Action-Observation 循环推理
2. Tool Calling Agent：使用原生 Function Calling API（推荐）
3. Structured Chat Agent：输出结构化 JSON
4. OpenAI Functions Agent：利用 OpenAI 特有的函数调用能力

生产环境推荐使用 Tool Calling Agent，因为它更可靠，错误率更低。""",
        metadata={"source": "agent_guide.txt", "section": "Agent类型"},
    ),
    Document(
        page_content="""LangChain 的 Memory 系统提供多种对话历史管理策略：
- ConversationBufferMemory：存储完整历史，适合短对话
- ConversationSummaryBufferMemory：超出 token 限制时自动压缩，生产推荐
- VectorStoreRetrieverMemory：语义检索历史，适合超长对话
- EntityMemory：追踪对话中提到的实体（人名、地名等）

LCEL 推荐使用 RunnableWithMessageHistory 替代旧版 ConversationChain。""",
        metadata={"source": "memory_guide.txt", "section": "Memory类型"},
    ),
]


def build_knowledge_base(docs: list[Document], cache_dir: str | None = None) -> FAISS:
    """
    构建向量知识库，使用 CacheBackedEmbeddings 缓存向量计算。
    
    Args:
        docs: 原始文档列表
        cache_dir: Embedding 缓存目录，None 时使用系统临时目录下的子目录
    
    Returns:
        FAISS 向量存储实例
    """
    import tempfile
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "langchain_embedding_cache")
    # 1. 文档分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "],
    )
    chunks = splitter.split_documents(docs)
    
    # 为每个 chunk 添加段落编号
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_count"] = len(chunks)
    
    print(f"文档分块完成：{len(docs)} 篇文档 → {len(chunks)} 个块")
    
    # 2. 配置带缓存的 Embedding（相同文本不重复调用 API）
    os.makedirs(cache_dir, exist_ok=True)
    store = LocalFileStore(cache_dir)
    
    base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=base_embeddings.model,  # 按模型名隔离缓存
    )
    
    # 3. 构建向量存储
    vectorstore = FAISS.from_documents(chunks, cached_embeddings)
    print(f"向量库构建完成，共 {vectorstore.index.ntotal} 个向量")
    
    return vectorstore


def build_rag_with_sources(vectorstore: FAISS):
    """
    构建返回来源信息的 RAG Chain。
    
    Returns:
        Callable: 接受 question 返回 {answer, sources} 的函数
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",               # MMR 减少冗余
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    # Prompt 要求模型标注引用
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库问答助手。严格基于以下上下文回答问题。

规则：
1. 只使用上下文中的信息，不要添加未提及的内容
2. 回答末尾必须列出引用来源，格式：[来源：文件名，章节]
3. 如果上下文不包含答案，回复"知识库中暂无相关信息"

---
{context}
---"""),
        ("human", "{question}"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 保存检索到的文档（用于返回来源信息）
    retrieved_docs: list[Document] = []
    
    def retrieve_and_store(question: str) -> str:
        """检索文档并存储，同时返回格式化上下文。"""
        nonlocal retrieved_docs
        retrieved_docs = retriever.invoke(question)
        return "\n\n".join(
            f"[文档{i+1} | 来源: {doc.metadata.get('source')} "
            f"| 章节: {doc.metadata.get('section')}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        )
    
    from langchain_core.runnables import RunnableLambda
    
    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda q: retrieve_and_store(q["question"])),
            question=RunnableLambda(lambda q: q["question"]),
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    def answer_with_sources(question: str) -> dict:
        """
        问答并返回答案与来源信息。
        
        Returns:
            dict: {
                "answer": str,
                "sources": [{"source": str, "section": str, "content": str}]
            }
        """
        answer = chain.invoke({"question": question})
        
        sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "section": doc.metadata.get("section", "未知"),
                "content": doc.page_content[:100] + "...",
            }
            for doc in retrieved_docs
        ]
        
        return {"answer": answer, "sources": sources}
    
    return answer_with_sources


def demo_rag():
    """演示 RAG 问答系统。"""
    print("=== 构建知识库 ===")
    vectorstore = build_knowledge_base(SAMPLE_DOCS)
    
    qa_func = build_rag_with_sources(vectorstore)
    
    questions = [
        "LangChain 中有哪些 Agent 类型？生产环境推荐使用哪种？",
        "如何在 LangChain 中管理长对话的历史记录？",
        "什么是 LCEL？有什么优势？",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        result = qa_func(q)
        print(f"\n回答:\n{result['answer']}")
        print(f"\n引用来源:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src['source']} / {src['section']}")
            print(f"     摘要: {src['content']}")


if __name__ == "__main__":
    demo_rag()
```

---

### 题目 2：实现 Self-Query 检索器（元数据过滤）

**要求：**
1. 文档包含元数据（时间、类别、作者）
2. 实现从自然语言问题中自动提取过滤条件（如"2024年的技术文章"）
3. 对比有/无元数据过滤的检索结果差异

---

### 题目 2 标准答案

```python
"""
题目2：元数据过滤检索器
知识点：结构化元数据、过滤检索、查询解析
"""
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


# 模拟带结构化元数据的文档
DOCS_WITH_METADATA = [
    Document(
        page_content="GPT-4 是 OpenAI 发布的多模态大语言模型，支持图像和文本输入。",
        metadata={"year": 2023, "category": "技术", "author": "OpenAI", "difficulty": "中级"},
    ),
    Document(
        page_content="LLaMA 3 是 Meta 发布的开源大语言模型，参数规模从 8B 到 70B。",
        metadata={"year": 2024, "category": "技术", "author": "Meta", "difficulty": "初级"},
    ),
    Document(
        page_content="AI 监管框架应包含透明度、可解释性和责任制三个核心原则。",
        metadata={"year": 2024, "category": "政策", "author": "EU", "difficulty": "高级"},
    ),
    Document(
        page_content="向量数据库 Milvus 2.4 发布，支持混合检索和稀疏向量。",
        metadata={"year": 2024, "category": "技术", "author": "Zilliz", "difficulty": "中级"},
    ),
    Document(
        page_content="大语言模型在医疗诊断中的应用：挑战与机遇。",
        metadata={"year": 2023, "category": "应用", "author": "Stanford", "difficulty": "高级"},
    ),
]


def build_self_query_retriever() -> SelfQueryRetriever:
    """
    构建支持自然语言元数据过滤的检索器。
    
    SelfQueryRetriever 会用 LLM 解析用户查询，自动提取：
    - 语义搜索内容（发送给向量检索器）
    - 结构化过滤条件（发送给元数据过滤器）
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(DOCS_WITH_METADATA, embeddings)
    
    # 描述元数据字段（LLM 需要这些信息来解析过滤条件）
    metadata_field_info = [
        AttributeInfo(
            name="year",
            description="文章发布年份",
            type="integer",
        ),
        AttributeInfo(
            name="category",
            description="文章类别，可选值：技术、政策、应用",
            type="string",
        ),
        AttributeInfo(
            name="author",
            description="文章作者或发布机构",
            type="string",
        ),
        AttributeInfo(
            name="difficulty",
            description="内容难度等级：初级、中级、高级",
            type="string",
        ),
    ]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="AI 和机器学习领域的技术文章",
        metadata_field_info=metadata_field_info,
        verbose=True,  # 显示 LLM 生成的结构化查询
        search_kwargs={"k": 3},
    )
    
    return retriever


def compare_retrieval(query: str, retriever: SelfQueryRetriever, vectorstore: Chroma):
    """对比有/无元数据过滤的检索结果。"""
    print(f"\n查询: {query}")
    print("-" * 40)
    
    # 有元数据过滤（SelfQueryRetriever）
    print("【SelfQuery 检索（自动过滤元数据）】")
    filtered_docs = retriever.invoke(query)
    for i, doc in enumerate(filtered_docs, 1):
        print(f"  {i}. [{doc.metadata['year']}年 | {doc.metadata['category']}] "
              f"{doc.page_content[:60]}...")
    
    # 普通语义搜索（无过滤）
    print("\n【普通语义搜索（无过滤）】")
    plain_docs = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(plain_docs, 1):
        print(f"  {i}. [{doc.metadata['year']}年 | {doc.metadata['category']}] "
              f"{doc.page_content[:60]}...")


if __name__ == "__main__":
    print("=== 构建 Self-Query 检索器 ===")
    retriever = build_self_query_retriever()
    
    # 需要单独创建 vectorstore 用于对比（实际中共用同一实例）
    vectorstore = Chroma.from_documents(DOCS_WITH_METADATA, OpenAIEmbeddings())
    
    # 测试自然语言元数据过滤
    compare_retrieval("2024年发布的技术文章", retriever, vectorstore)
    compare_retrieval("高级难度的政策相关内容", retriever, vectorstore)
```

---

## 三、知识总结

### RAG 质量优化清单

```
【检索质量】
✅ chunk_size 和 overlap 根据文档类型调优
✅ 使用 MMR 减少检索结果冗余
✅ Hybrid Search（BM25 + 语义）提升召回率
✅ Re-ranking（交叉编码器）提升精度
✅ 元数据过滤缩小搜索范围

【生成质量】
✅ Prompt 明确要求基于上下文回答
✅ 检索结果包含来源信息
✅ 温度设为 0（减少幻觉）
✅ 实现 Faithfulness 评估
```

### 面试高频问题

1. **RAG 和 Fine-tuning 如何选择？**
   - 知识更新频繁、数据私有 → RAG；需要改变模型行为/风格 → Fine-tuning

2. **如何评估 RAG 系统质量？**
   - 检索相关性（Recall@K）、答案忠实度（Faithfulness）、答案相关性（Answer Relevance）

3. **chunk_overlap 的作用是什么？**
   - 防止重要信息被分割在两个块的边界，保持语义连续性
