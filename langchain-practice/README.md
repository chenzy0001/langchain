# LangChain Agent 工程能力学习指南

> **面向 AI Agent 方向面试的系统性学习材料**  
> 涵盖理论教学、代码实践、题目与标准答案、简历话术

---

## 目录

| 章节 | 主题 | 核心价值 |
|------|------|---------|
| [01](./01_react_agent/) | ReAct Agent 模式 | Agent 核心推理范式，面试必考 |
| [02](./02_tool_use/) | 工具设计与管理 | Agent 能力扩展的关键 |
| [03](./03_memory/) | 记忆与上下文管理 | 长对话、多轮交互的核心 |
| [04](./04_rag_knowledge_base/) | RAG 知识库检索增强 | 企业级 Agent 最常见架构 |
| [05](./05_lcel_custom_agent/) | LCEL 自定义 Agent | 现代 LangChain 构建范式 |
| [06](./06_multi_agent/) | 多 Agent 协作系统 | 复杂任务分解与协作 |
| [07](./07_streaming_async/) | 流式输出与异步处理 | 生产级性能优化 |
| [08](./08_caching_strategies/) | 缓存策略 | 降本提速，工程化必备 |

---

## 学习路径建议

```
Week 1: 01 ReAct → 02 Tool Use → 03 Memory
Week 2: 04 RAG  → 05 LCEL Custom Agent
Week 3: 06 Multi-Agent → 07 Streaming → 08 Caching
```

---

## 简历技能话术（直接可用）

以下话术经过提炼，既有技术深度又简洁，适合写在简历 **项目经历** 或 **技术技能** 栏：

### 1. Agent 核心架构
> 基于 LangChain ReAct 框架设计并实现 AI Agent，通过 Thought-Action-Observation 循环实现复杂多步推理，支持工具调用（Tool Calling）与自主决策，在 [具体业务场景] 中将任务完成率提升至 XX%。

### 2. 工具系统工程
> 设计并实现可扩展的 Agent 工具生态，包括自定义结构化工具（Structured Tool）、工具路由策略与错误恢复机制；使用 OpenAI Function Calling 标准规范工具输入输出 Schema，确保工具调用的可靠性与可观测性。

### 3. 记忆与上下文管理
> 设计多层次记忆架构：使用 ConversationSummaryBufferMemory 在 token 限制内保留最大信息量，结合 VectorStore-backed Memory 实现语义检索式历史记忆，支持跨会话持久化存储。

### 4. RAG 知识库增强检索
> 构建生产级 RAG Pipeline：文档分块（RecursiveCharacterTextSplitter）→ 向量化（Embeddings）→ 向量数据库检索（FAISS/Chroma/Pinecone）→ 上下文注入，实现领域知识问答；通过 Re-ranking 与 Hybrid Search 将检索准确率提升 XX%。

### 5. LCEL 声明式 Agent 构建
> 熟练使用 LangChain Expression Language（LCEL）构建可组合的 Agent Pipeline，利用 `|` 操作符链接 Prompt → LLM → OutputParser，实现流式输出、批处理、并行执行与异步调用，降低延迟 XX ms。

### 6. 多 Agent 协作系统
> 设计并实现 Supervisor-Worker 多 Agent 架构，Supervisor Agent 负责任务规划与分发，多个专职 Worker Agent 并行执行子任务，最终聚合结果；系统支持动态路由与错误降级，适用于复杂企业工作流自动化。

### 7. 流式与异步优化
> 使用 LangChain `astream()` 和 `astream_events()` 实现 Server-Sent Events（SSE）流式响应，显著降低首 Token 延迟（TTFT）；结合 `asyncio` 实现并发 Agent 调用，将批量任务吞吐量提升 XX 倍。

### 8. 成本与性能优化
> 实施多级缓存策略：LLM 语义缓存（`GPTCache` / `InMemoryCache`）+ Embedding 缓存（`CacheBackedEmbeddings`），在高频查询场景下将 API 调用成本降低 60%+，平均响应时间从 XX 秒降至 XX 毫秒。

---

## 推荐学习资源

| 资源 | 链接 | 说明 |
|------|------|------|
| LangChain 官方文档 | https://python.langchain.com/docs/ | 最权威参考 |
| LangChain GitHub | https://github.com/langchain-ai/langchain | 源码学习 |
| LangGraph 文档 | https://langchain-ai.github.io/langgraph/ | 多 Agent 最新框架 |
| LangSmith | https://smith.langchain.com/ | Agent 可观测性平台 |
| ReAct Paper | https://arxiv.org/abs/2210.03629 | 理论基础 |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | Transformer 基础 |
| OpenAI Function Calling | https://platform.openai.com/docs/guides/function-calling | 工具调用规范 |

---

## 考试题目总览

每个章节均包含 1~2 道练习题及标准答案，题目设计原则：
- **不照抄源码**，但体现核心工程能力
- 覆盖设计、实现、调试、优化四个维度
- 每题均有详细解析与评分要点

进入各章节目录查看详情。
