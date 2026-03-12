# 03 记忆与上下文管理（Memory Management）

## 一、理论教学

### 3.1 为什么需要 Memory？

LLM 本质上是**无状态**的：每次 API 调用都是独立的，不记得之前的对话。Memory 模块的作用是：
1. 将历史对话存储起来
2. 在下次对话时，将相关历史注入 Prompt
3. 在 Token 限制内，最大化保留有用的上下文

### 3.2 Memory 的类型对比

| Memory 类型 | 存储方式 | 适用场景 | Token 消耗 |
|------------|---------|---------|-----------|
| `ConversationBufferMemory` | 完整对话历史 | 短对话、需要精确回溯 | 随轮数线性增长 |
| `ConversationBufferWindowMemory` | 最近 K 轮 | 固定窗口的轻量场景 | 固定上限 |
| `ConversationSummaryMemory` | LLM 生成的摘要 | 长对话、不需要精确词句 | 大幅压缩 |
| `ConversationSummaryBufferMemory` | 摘要 + 最近完整对话 | **生产推荐**，平衡精度与成本 | 可控 |
| `VectorStoreRetrieverMemory` | 向量数据库，语义检索 | 超长历史、需要语义搜索 | 按相关性截断 |
| `EntityMemory` | 实体信息字典 | 需要追踪人名/地名/概念 | 按实体数量 |

### 3.3 架构设计：多层次记忆系统

生产级 Agent 通常需要组合多种 Memory：

```
┌──────────────────────────────────────────────┐
│               用户请求                        │
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Working Memory   │  ← 当前对话（ConversationSummaryBufferMemory）
         │   (短期记忆)        │
         └─────────┬──────────┘
                   │ 超过阈值时
         ┌─────────▼──────────┐
         │   Episodic Memory  │  ← 历史会话摘要（VectorStore）
         │   (情节记忆)        │
         └─────────┬──────────┘
                   │ 长期保存
         ┌─────────▼──────────┐
         │   Semantic Memory  │  ← 用户偏好/实体信息（Key-Value Store）
         │   (语义记忆)        │
         └────────────────────┘
```

### 3.4 ConversationSummaryBufferMemory 原理

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # 超过此 token 数触发摘要压缩
    return_messages=True,  # 返回 Message 对象（而非字符串）
    memory_key="chat_history",
    output_key="output",   # Agent 响应的 key
)

# 内部逻辑：
# 1. 新对话先追加到 buffer
# 2. 当 buffer 超过 max_token_limit 时：
#    a. 用 LLM 对最旧的消息生成摘要
#    b. 摘要替换原始消息，释放 Token 空间
#    c. 最新的几轮对话保持完整形式
```

### 3.5 LCEL 中的 Memory 集成

现代 LangChain 推荐使用 `RunnableWithMessageHistory` 管理对话历史：

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 会话存储（生产中使用 Redis / PostgreSQL）
session_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取或创建对话历史。"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# 构建带历史的 Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的 AI 助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息注入点
    ("human", "{input}"),
])

chain = prompt | ChatOpenAI(model="gpt-4o-mini")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 同一 session_id 的调用共享历史
response1 = chain_with_history.invoke(
    {"input": "我叫张三"},
    config={"configurable": {"session_id": "user_001"}}
)

response2 = chain_with_history.invoke(
    {"input": "你记得我叫什么名字吗？"},
    config={"configurable": {"session_id": "user_001"}}
)
# Response2 会正确回答"张三"
```

### 3.6 持久化记忆：跨会话存储

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_session_history(session_id: str) -> RedisChatMessageHistory:
    """使用 Redis 持久化对话历史，支持跨进程/跨服务共享。"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=86400,  # 24 小时过期
        key_prefix="langchain:chat:",
    )
```

---

## 二、实践练习

### 题目 1：实现智能 Token 预算管理的 Memory

**背景：** 在生产环境中，控制 Memory 的 Token 消耗是降低成本的关键。

**要求：**
1. 实现一个 `BudgetedMemory` 类，支持设置最大 Token 预算
2. 当历史消息超过预算时，自动压缩（策略：保留最近 N 轮 + 对早期消息生成摘要）
3. 提供 `token_usage()` 方法返回当前 Token 使用情况
4. 与 `RunnableWithMessageHistory` 集成

---

### 题目 1 标准答案

```python
"""
题目1：带 Token 预算的智能 Memory
知识点：Chat History、Token 管理、摘要压缩、LCEL 集成
"""
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def count_tokens_approx(messages: list[BaseMessage]) -> int:
    """粗略估算消息的 Token 数（4字符≈1 Token）。"""
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


class BudgetedMemory(BaseChatMessageHistory):
    """
    带 Token 预算的对话历史管理器。
    
    当历史消息超过预算时，自动压缩早期消息为摘要，
    同时保留最近的完整对话轮次。
    """
    
    def __init__(
        self,
        max_token_budget: int = 2000,
        keep_recent_turns: int = 3,
        summary_llm: ChatOpenAI = None,
    ):
        """
        Args:
            max_token_budget: 最大 Token 预算
            keep_recent_turns: 始终保持完整形式的最近轮次数
            summary_llm: 用于生成摘要的 LLM
        """
        self.max_token_budget = max_token_budget
        self.keep_recent_turns = keep_recent_turns
        self._summary_llm = summary_llm or ChatOpenAI(model="gpt-4o-mini")
        self._messages: list[BaseMessage] = []
        self._summary: str = ""  # 早期对话的摘要
        self._compression_count: int = 0
    
    @property
    def messages(self) -> list[BaseMessage]:
        """返回供 Prompt 使用的消息列表（摘要 + 最近完整对话）。"""
        result = []
        if self._summary:
            result.append(SystemMessage(content=f"[之前对话摘要]: {self._summary}"))
        result.extend(self._messages)
        return result
    
    def add_message(self, message: BaseMessage) -> None:
        """添加新消息，必要时触发压缩。"""
        self._messages.append(message)
        
        # 检查是否需要压缩
        current_tokens = count_tokens_approx(self.messages)
        if current_tokens > self.max_token_budget:
            self._compress()
    
    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))
    
    def _compress(self) -> None:
        """压缩早期消息：保留最近 N 轮，其余生成摘要。"""
        # 最近 N 轮 = N*2 条消息（每轮 1 Human + 1 AI）
        keep_count = self.keep_recent_turns * 2
        
        if len(self._messages) <= keep_count:
            return  # 消息不够多，无需压缩
        
        # 分割：待压缩的早期消息 vs 保留的最近消息
        to_compress = self._messages[:-keep_count]
        self._messages = self._messages[-keep_count:]
        
        # 生成摘要
        existing_summary = f"之前摘要：{self._summary}\n\n" if self._summary else ""
        new_content = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in to_compress
        )
        
        prompt = (
            f"{existing_summary}"
            f"请用中文简洁地总结以下对话内容，保留关键信息（人名、决策、重要事实）：\n\n"
            f"{new_content}"
        )
        
        summary_response = self._summary_llm.invoke(prompt)
        self._summary = summary_response.content
        self._compression_count += 1
        
        print(f"[Memory] 压缩完成（第{self._compression_count}次），"
              f"当前 Token: {count_tokens_approx(self.messages)}")
    
    def token_usage(self) -> dict:
        """返回当前 Token 使用情况。"""
        current = count_tokens_approx(self.messages)
        return {
            "current_tokens": current,
            "max_budget": self.max_token_budget,
            "usage_rate": f"{current / self.max_token_budget:.1%}",
            "message_count": len(self._messages),
            "has_summary": bool(self._summary),
            "compression_count": self._compression_count,
        }
    
    def clear(self) -> None:
        """清空历史。"""
        self._messages = []
        self._summary = ""
        self._compression_count = 0


# ===== 集成 RunnableWithMessageHistory =====

memory_store: dict[str, BudgetedMemory] = {}

def get_budgeted_memory(session_id: str) -> BudgetedMemory:
    """获取或创建带预算的 Memory 实例。"""
    if session_id not in memory_store:
        memory_store[session_id] = BudgetedMemory(
            max_token_budget=500,  # 演示用，实际可设 2000-4000
            keep_recent_turns=2,
        )
    return memory_store[session_id]


def create_budgeted_chat_chain():
    """创建带 Token 预算管理的对话链。"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的 AI 助手，记住用户提到的重要信息。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return RunnableWithMessageHistory(
        chain,
        get_budgeted_memory,
        input_messages_key="input",
        history_messages_key="history",
    )


def demo_budgeted_memory():
    """演示 Token 预算管理效果。"""
    chain = create_budgeted_chat_chain()
    session_id = "test_session"
    config = {"configurable": {"session_id": session_id}}
    
    conversations = [
        "我叫张三，是一名 Python 工程师，在北京工作。",
        "我正在研究 LangChain，想用它构建一个 RAG 系统。",
        "我们公司的向量数据库用的是 Milvus，你了解吗？",
        "除了 Milvus，还有哪些向量数据库值得考虑？",
        "对了，我之前说我在哪个城市工作？",  # 测试记忆
    ]
    
    for user_input in conversations:
        print(f"\n用户: {user_input}")
        response = chain.invoke({"input": user_input}, config=config)
        print(f"AI: {response}")
        
        # 显示 Token 使用情况
        memory = get_budgeted_memory(session_id)
        usage = memory.token_usage()
        print(f"[Memory状态] Token使用: {usage['current_tokens']}/{usage['max_budget']} "
              f"({usage['usage_rate']}), 摘要: {'有' if usage['has_summary'] else '无'}")


if __name__ == "__main__":
    demo_budgeted_memory()
```

---

### 题目 2：实现跨会话的持久化 Memory（SQLite 版）

**要求：**
1. 使用 SQLite 实现 `BaseChatMessageHistory`，跨进程持久化对话历史
2. 支持按 `session_id` 隔离不同用户的对话
3. 实现 `get_session_stats()` 返回所有活跃会话的统计信息

---

### 题目 2 标准答案

```python
"""
题目2：SQLite 持久化 Memory
知识点：自定义 ChatMessageHistory、数据持久化、会话管理
"""
import sqlite3
import json
import os
from datetime import datetime
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage,
    SystemMessage, messages_from_dict, messages_to_dict
)


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """
    基于 SQLite 的持久化对话历史。
    
    特性：
    - 跨进程持久化
    - 按 session_id 隔离
    - 支持消息类型（Human/AI/System）
    """
    
    def __init__(self, session_id: str, db_path: str = "chat_history.db"):
        """
        Args:
            session_id: 会话唯一标识
            db_path: SQLite 数据库文件路径
        """
        self.session_id = session_id
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    CONSTRAINT valid_type CHECK (
                        message_type IN ('human', 'ai', 'system', 'function', 'tool')
                    )
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages(session_id)"
            )
            conn.commit()
    
    @property
    def messages(self) -> list[BaseMessage]:
        """从数据库加载当前会话的所有消息。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT message_type, content, metadata FROM chat_messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,)
            ).fetchall()
        
        message_dicts = [
            {
                "type": row["message_type"],
                "data": {
                    "content": row["content"],
                    "additional_kwargs": json.loads(row["metadata"]),
                }
            }
            for row in rows
        ]
        return messages_from_dict(message_dicts)
    
    def add_message(self, message: BaseMessage) -> None:
        """持久化存储新消息。"""
        type_map = {
            HumanMessage: "human",
            AIMessage: "ai",
            SystemMessage: "system",
        }
        # 对于未在映射中的消息类型（如 FunctionMessage、ToolMessage），
        # 通过 type 属性获取其字符串类型，默认退回 "ai"
        message_type = type_map.get(type(message)) or getattr(message, "type", "ai")
        # 确保类型值在约束允许范围内
        if message_type not in ("human", "ai", "system", "function", "tool"):
            message_type = "ai"
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_messages "
                "(session_id, message_type, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    message_type,
                    message.content,
                    json.dumps(message.additional_kwargs),
                    datetime.now().isoformat(),
                )
            )
            conn.commit()
    
    def clear(self) -> None:
        """清除当前会话的所有消息。"""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?",
                (self.session_id,)
            )
            conn.commit()
    
    @classmethod
    def get_session_stats(cls, db_path: str = "chat_history.db") -> list[dict]:
        """获取所有活跃会话的统计信息（类方法）。"""
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        rows = conn.execute("""
            SELECT 
                session_id,
                COUNT(*) as message_count,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message,
                SUM(CASE WHEN message_type = 'human' THEN 1 ELSE 0 END) as human_count,
                SUM(CASE WHEN message_type = 'ai' THEN 1 ELSE 0 END) as ai_count
            FROM chat_messages
            GROUP BY session_id
            ORDER BY last_message DESC
        """).fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]


def demo_sqlite_memory():
    """演示 SQLite 持久化 Memory。"""
    import tempfile
    
    # 使用临时文件避免污染当前目录
    db_path = os.path.join(tempfile.gettempdir(), "demo_chat.db")
    
    # 模拟两个用户的对话
    for user_id in ["user_alice", "user_bob"]:
        history = SQLiteChatMessageHistory(session_id=user_id, db_path=db_path)
        history.add_user_message(f"你好，我是{user_id.split('_')[1]}")
        history.add_ai_message(f"你好！很高兴认识你，{user_id.split('_')[1]}！")
        history.add_user_message("帮我解释一下什么是 LangChain")
        history.add_ai_message("LangChain 是一个用于构建 AI 应用的框架...")
    
    # 验证持久化
    print("=== Alice 的对话历史 ===")
    alice_history = SQLiteChatMessageHistory(session_id="user_alice", db_path=db_path)
    for msg in alice_history.messages:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}]: {msg.content}")
    
    # 查看所有会话统计
    print("\n=== 所有会话统计 ===")
    stats = SQLiteChatMessageHistory.get_session_stats(db_path)
    for s in stats:
        print(f"  会话 {s['session_id']}: {s['message_count']} 条消息"
              f"（用户 {s['human_count']} / AI {s['ai_count']}）")
    
    # 清理临时文件
    os.remove(db_path)


if __name__ == "__main__":
    demo_sqlite_memory()
```

---

## 三、知识总结

### Memory 选型决策树

```
需要精确记住每句话？
├── 是 → 对话较短？
│   ├── 是 → ConversationBufferMemory
│   └── 否 → ConversationSummaryBufferMemory（生产首选）
└── 否 → 需要语义搜索历史？
    ├── 是 → VectorStoreRetrieverMemory
    └── 否 → ConversationSummaryMemory
```

### 面试高频问题

1. **如何解决 LLM 上下文窗口限制问题？**
   - 摘要压缩（SummaryMemory）、滑动窗口（WindowMemory）、向量检索相关历史

2. **多用户场景如何隔离 Memory？**
   - 使用 `session_id` 作为 Key，每个用户独立的 `ChatMessageHistory` 实例

3. **`RunnableWithMessageHistory` 相比旧 `ConversationChain` 的优势？**
   - 支持 LCEL 组合、显式 session 管理、更好的异步支持、与 LangGraph 兼容
