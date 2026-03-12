# 07 流式输出与异步处理

## 一、理论教学

### 7.1 为什么需要流式输出？

| 指标 | 非流式 | 流式 |
|------|--------|------|
| 用户体验 | 等待完整响应（卡顿感） | 逐字显示（流畅感） |
| TTFT（首 Token 时间） | = 总生成时间 | < 1 秒 |
| 前端实现 | 简单轮询 | SSE / WebSocket |
| 超时风险 | 高（长响应易超时） | 低（持续传输）|

**关键概念：**
- **TTFT（Time to First Token）**：用户从发起请求到看到第一个字的时间
- **SSE（Server-Sent Events）**：HTTP 长连接，服务器主动推送事件，适合 AI 流式输出
- **Token 速率**：LLM 每秒生成的 Token 数，影响流式效果

### 7.2 LCEL 中的流式接口

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("解释{topic}") | ChatOpenAI() | StrOutputParser()

# 方法1：stream - 同步流式
for chunk in chain.stream({"topic": "量子计算"}):
    print(chunk, end="", flush=True)

# 方法2：astream - 异步流式
async def astream_demo():
    async for chunk in chain.astream({"topic": "量子计算"}):
        print(chunk, end="", flush=True)

# 方法3：astream_events - 细粒度事件流（LangChain v0.2+）
async def astream_events_demo():
    async for event in chain.astream_events(
        {"topic": "量子计算"},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        elif event["event"] == "on_tool_start":
            print(f"\n[调用工具: {event['name']}]")
        elif event["event"] == "on_tool_end":
            print(f"[工具返回: {event['data']['output'][:50]}]")
```

### 7.3 astream_events 事件类型

`astream_events` 是 LangChain 最强大的流式接口，可以监听整个 Chain 中的所有事件：

| 事件名 | 触发时机 | 关键数据 |
|--------|---------|---------|
| `on_chain_start` | Chain 开始执行 | `inputs` |
| `on_chain_end` | Chain 执行完毕 | `outputs` |
| `on_chat_model_start` | LLM 开始生成 | `messages` |
| `on_chat_model_stream` | LLM 每输出一个 Token | `chunk.content` |
| `on_chat_model_end` | LLM 生成完毕 | `response` |
| `on_tool_start` | 工具开始执行 | `input` |
| `on_tool_end` | 工具执行完毕 | `output` |
| `on_retriever_start` | 检索器开始 | `query` |
| `on_retriever_end` | 检索器完成 | `documents` |

### 7.4 异步并发：提升 Agent 吞吐量

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# ❌ 串行调用（慢）
async def serial_calls(questions: list[str]) -> list[str]:
    results = []
    for q in questions:  # 顺序执行，总时间 = 每次调用时间之和
        result = await llm.ainvoke(q)
        results.append(result.content)
    return results

# ✅ 并发调用（快）
async def parallel_calls(questions: list[str]) -> list[str]:
    tasks = [llm.ainvoke(q) for q in questions]  # 创建协程
    results = await asyncio.gather(*tasks)  # 并发执行
    return [r.content for r in results]

# ✅ 使用 batch（LangChain 内建并发）
def batch_calls(questions: list[str]) -> list[str]:
    results = llm.batch(questions, config={"max_concurrency": 5})
    return [r.content for r in results]
```

### 7.5 FastAPI + SSE 流式 API

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

app = FastAPI()

chain = (
    ChatPromptTemplate.from_template("回答：{question}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

@app.get("/stream")
async def stream_response(question: str):
    """SSE 流式响应接口。"""
    
    async def event_generator():
        async for chunk in chain.astream({"question": question}):
            # SSE 格式：data: <content>\n\n
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲
        }
    )
```

---

## 二、实践练习

### 题目 1：实现 Agent 的实时进度推送

**背景：** 用户调用 Agent 处理复杂任务时，需要实时看到每个步骤的进度（工具调用、当前状态等），而不是等待最终结果。

**要求：**
1. 使用 `astream_events` 实现 Agent 执行过程的实时事件推送
2. 区分并格式化以下事件：工具调用开始/结束、LLM 思考中、最终答案
3. 实现一个 `async` 生成器函数 `stream_agent_events`，可被 FastAPI/WebSocket 消费

---

### 题目 1 标准答案

```python
"""
题目1：Agent 实时进度推送
知识点：astream_events、事件过滤、AsyncGenerator
"""
import asyncio
import json
from typing import AsyncGenerator
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 定义测试工具
@tool
async def async_search(query: str) -> str:
    """搜索相关信息。Args: query: 搜索关键词"""
    await asyncio.sleep(0.5)  # 模拟异步 API 调用
    return f"搜索结果[{query}]: 找到相关内容 3 条..."


@tool
async def async_calculate(expression: str) -> str:
    """计算简单数学表达式（仅支持 +、-、*、/ 运算）。Args: expression: 数学表达式"""
    import ast
    import operator as op
    
    await asyncio.sleep(0.2)
    
    allowed_ops = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        msg = f"不支持的表达式: {type(node).__name__}"
        raise ValueError(msg)
    
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = safe_eval(tree.body)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


def create_async_agent() -> AgentExecutor:
    """创建支持异步的 Agent。"""
    tools = [async_search, async_calculate]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，能搜索信息和执行计算。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


async def stream_agent_events(
    executor: AgentExecutor,
    question: str,
) -> AsyncGenerator[dict, None]:
    """
    流式推送 Agent 执行事件。
    
    Yields:
        dict: 格式化的事件字典，包含 type、content 等字段
    
    事件类型：
        - thinking: LLM 正在生成内容
        - tool_start: 开始调用工具
        - tool_end: 工具调用完成  
        - answer: 最终答案
        - error: 发生错误
    """
    yield {"type": "start", "content": "Agent 开始处理...", "step": 0}
    
    step_count = 0
    
    try:
        async for event in executor.astream_events(
            {"input": question},
            version="v2",
        ):
            event_name = event.get("event", "")
            event_data = event.get("data", {})
            tags = event.get("tags", [])
            
            # 过滤：只处理顶层 Chain 的事件（忽略嵌套子 Chain）
            if "seq:step:" in str(tags):
                continue
            
            if event_name == "on_chat_model_stream":
                # LLM 逐 Token 输出
                chunk = event_data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "thinking",
                        "content": chunk.content,
                        "step": step_count,
                    }
            
            elif event_name == "on_tool_start":
                step_count += 1
                tool_name = event.get("name", "unknown")
                tool_input = event_data.get("input", {})
                yield {
                    "type": "tool_start",
                    "content": f"正在调用工具: {tool_name}",
                    "tool": tool_name,
                    "input": str(tool_input)[:100],
                    "step": step_count,
                }
            
            elif event_name == "on_tool_end":
                tool_output = event_data.get("output", "")
                yield {
                    "type": "tool_end",
                    "content": f"工具返回: {str(tool_output)[:100]}",
                    "output": str(tool_output)[:200],
                    "step": step_count,
                }
            
            elif event_name == "on_chain_end":
                output = event_data.get("output", {})
                if isinstance(output, dict) and "output" in output:
                    yield {
                        "type": "answer",
                        "content": output["output"],
                        "step": step_count,
                    }
    
    except Exception as e:
        yield {"type": "error", "content": f"Agent 执行出错: {e}", "step": step_count}
    
    yield {"type": "done", "content": "执行完毕", "step": step_count}


async def demo_stream_events():
    """演示 Agent 实时进度推送。"""
    executor = create_async_agent()
    question = "搜索LangChain最新特性，并计算 2024 + 365"
    
    print(f"问题: {question}\n{'='*50}")
    
    last_type = None
    async for event in stream_agent_events(executor, question):
        event_type = event["type"]
        
        # 格式化输出
        if event_type == "start":
            print(f"🚀 {event['content']}")
        
        elif event_type == "thinking":
            if last_type != "thinking":
                print("\n💭 思考中: ", end="")
            print(event["content"], end="", flush=True)
        
        elif event_type == "tool_start":
            print(f"\n\n🔧 [{event['step']}] {event['content']}")
            print(f"   输入: {event['input']}")
        
        elif event_type == "tool_end":
            print(f"   ✓ {event['content']}")
        
        elif event_type == "answer":
            print(f"\n\n✅ 最终答案:\n{event['content']}")
        
        elif event_type == "done":
            print(f"\n{'='*50}")
            print(f"🏁 完成！共执行 {event['step']} 次工具调用")
        
        elif event_type == "error":
            print(f"\n❌ 错误: {event['content']}")
        
        last_type = event_type


# ===== FastAPI 集成示例 =====
FASTAPI_EXAMPLE = '''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
executor = create_async_agent()

@app.get("/agent/stream")
async def agent_stream(question: str):
    """Agent 流式执行 API（SSE 格式）。"""
    
    async def event_generator():
        async for event in stream_agent_events(executor, question):
            # SSE 格式
            yield f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
'''


if __name__ == "__main__":
    asyncio.run(demo_stream_events())
    print("\n=== FastAPI 集成示例 ===")
    print(FASTAPI_EXAMPLE)
```

---

### 题目 2：实现高并发 Agent 批处理

**要求：**
1. 实现 `batch_process` 函数，并发处理多个用户查询
2. 支持设置最大并发数（避免 API 限流）
3. 统计每个查询的耗时和总体吞吐量（queries/second）
4. 错误的查询不影响其他查询的执行

---

### 题目 2 标准答案

```python
"""
题目2：高并发 Agent 批处理
知识点：asyncio.gather、信号量（Semaphore）、并发控制、性能统计
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class ProcessResult:
    """单条查询的处理结果。"""
    query: str
    output: Optional[str]
    error: Optional[str]
    latency_ms: float
    success: bool


async def process_single_query(
    chain,
    query: str,
    semaphore: asyncio.Semaphore,
) -> ProcessResult:
    """
    处理单条查询，受信号量限制并发数。
    
    Args:
        chain: LangChain Runnable
        query: 用户查询
        semaphore: 并发控制信号量
    """
    start = time.time()
    
    async with semaphore:  # 进入临界区（受并发数限制）
        try:
            output = await chain.ainvoke({"question": query})
            latency = (time.time() - start) * 1000
            return ProcessResult(
                query=query,
                output=output,
                error=None,
                latency_ms=round(latency, 1),
                success=True,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ProcessResult(
                query=query,
                output=None,
                error=str(e),
                latency_ms=round(latency, 1),
                success=False,
            )


async def batch_process(
    queries: list[str],
    max_concurrency: int = 5,
) -> dict:
    """
    高并发批处理多个查询。
    
    Args:
        queries: 查询列表
        max_concurrency: 最大并发数（防止 API 限流）
    
    Returns:
        dict: {
            results: List[ProcessResult],
            stats: {total, success, failed, avg_latency_ms, throughput_qps}
        }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = (
        ChatPromptTemplate.from_template("简洁地回答（50字以内）：{question}")
        | llm
        | StrOutputParser()
    )
    
    # 信号量：限制同时执行的协程数
    semaphore = asyncio.Semaphore(max_concurrency)
    
    print(f"开始批处理: {len(queries)} 条查询，最大并发: {max_concurrency}")
    
    total_start = time.time()
    
    # 创建所有任务并并发执行
    tasks = [
        process_single_query(chain, query, semaphore)
        for query in queries
    ]
    
    results: list[ProcessResult] = await asyncio.gather(*tasks)
    
    total_time = time.time() - total_start
    
    # 统计
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    throughput = len(queries) / total_time if total_time > 0 else 0
    
    stats = {
        "total": len(queries),
        "success": len(successful),
        "failed": len(failed),
        "avg_latency_ms": round(avg_latency, 1),
        "total_time_s": round(total_time, 2),
        "throughput_qps": round(throughput, 2),
    }
    
    return {"results": results, "stats": stats}


async def demo_batch_processing():
    """演示批处理性能对比。"""
    queries = [
        "Python 和 Java 的主要区别是什么？",
        "解释什么是 Docker 容器",
        "什么是 REST API？",
        "机器学习和深度学习的区别？",
        "解释微服务架构",
        "什么是 CI/CD？",
        "解释 SQL 和 NoSQL 的区别",
        "什么是负载均衡？",
    ]
    
    # 测试不同并发数的性能差异
    for max_concurrency in [1, 4]:
        print(f"\n{'='*50}")
        print(f"并发数: {max_concurrency}")
        
        result = await batch_process(queries[:4], max_concurrency=max_concurrency)
        stats = result["stats"]
        
        print(f"总耗时: {stats['total_time_s']}s")
        print(f"平均延迟: {stats['avg_latency_ms']}ms")
        print(f"吞吐量: {stats['throughput_qps']} QPS")
        print(f"成功/失败: {stats['success']}/{stats['failed']}")
        
        print("\n各查询耗时:")
        for r in result["results"]:
            status = "✅" if r.success else "❌"
            print(f"  {status} [{r.latency_ms}ms] {r.query[:30]}...")
    
    # 演示错误不影响其他查询
    print(f"\n{'='*50}")
    print("演示错误隔离（部分查询会失败）:")
    mixed_queries = ["正常查询1", "", "正常查询2", "   ", "正常查询3"]
    result = await batch_process(mixed_queries, max_concurrency=3)
    print(f"结果: {result['stats']}")


if __name__ == "__main__":
    asyncio.run(demo_batch_processing())
```

---

## 三、知识总结

### 流式与异步最佳实践

```python
# ✅ 生产环境推荐配置
llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,          # 开启流式
    max_retries=3,           # 自动重试
    request_timeout=30,      # 单次超时
)

# ✅ 批处理并发控制
semaphore = asyncio.Semaphore(5)  # 根据 API 限流阈值设置

# ✅ 使用 astream_events 而非 astream（更多信息）
async for event in chain.astream_events(input, version="v2"):
    ...
```

### 面试高频问题

1. **TTFT 和总延迟的区别？如何优化 TTFT？**
   - TTFT = 请求到第一个 Token 的时间；使用流式输出可以显著降低 TTFT

2. **`asyncio.gather` 和 `asyncio.Semaphore` 的配合使用场景？**
   - gather 实现并发，Semaphore 限制最大并发数防止 API 限流

3. **astream_events 相比 astream 有什么优势？**
   - astream 只返回最终输出的流；astream_events 返回整个 Pipeline 的所有中间事件
