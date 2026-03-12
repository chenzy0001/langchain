# 01 ReAct Agent 模式

## 一、理论教学

### 1.1 什么是 ReAct？

**ReAct**（Reasoning + Acting）是目前最主流的 AI Agent 推理范式，由 Google 在 2022 年提出（论文：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)）。

其核心思想是：让 LLM 在每一步交替进行 **思考（Thought）** 和 **行动（Action）**，并根据行动的 **观察（Observation）** 结果继续推理，直到得出最终答案。

```
问题输入
  ↓
Thought: 我需要先查询天气...
Action: search_weather(city="北京")
Observation: 北京今天晴，26°C
  ↓
Thought: 天气适合出行，现在查交通...
Action: search_traffic(route="北京→上海")
Observation: 高铁 4h30min，票价 ¥553
  ↓
Thought: 信息足够了，可以回答用户
Final Answer: 推荐乘坐高铁，今日天气晴好，票价 553 元
```

### 1.2 为什么这样设计？

| 设计选择 | 原因 |
|---------|------|
| 交替 Thought/Action | 避免 LLM 一次性"幻想"答案，强制验证每步结论 |
| Observation 反馈 | 将真实世界状态注入 LLM 上下文，减少错误传播 |
| 循环直到 Final Answer | 支持不确定步数的复杂任务，无需预定义工作流 |
| 工具封装为 Action | 解耦 LLM 推理与外部系统，便于替换和扩展 |

### 1.3 LangChain 中的 ReAct 架构

```
┌─────────────────────────────────────────┐
│              AgentExecutor              │
│  ┌─────────────┐    ┌────────────────┐  │
│  │    Agent    │───▶│  Tool Router   │  │
│  │ (LLM+Prompt)│◀───│  (工具分发器)   │  │
│  └─────────────┘    └────────────────┘  │
│         ▲                  │            │
│         │ Observation      │ Action     │
│         │                  ▼            │
│  ┌──────┴──────────────────────────┐   │
│  │         Tool Registry           │   │
│  │  [Calculator] [Search] [DB] ... │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 1.4 核心源码解析

LangChain 中 `AgentExecutor` 的核心循环逻辑（简化版）：

```python
# 源码参考: libs/langchain/langchain_classic/agents/agent.py
class AgentExecutor:
    def _call(self, inputs):
        intermediate_steps = []  # 存储 (Action, Observation) 历史
        
        while True:
            # 1. Agent 根据当前状态决策下一步行动
            output = self.agent.plan(
                intermediate_steps=intermediate_steps,
                **inputs
            )
            
            # 2. 如果输出是 AgentFinish，结束循环
            if isinstance(output, AgentFinish):
                return output.return_values
            
            # 3. 如果输出是 AgentAction，执行工具
            action = output  # AgentAction(tool, tool_input, log)
            observation = self._execute_tool(action)
            
            # 4. 将 (行动, 观察) 追加到历史，进入下一轮
            intermediate_steps.append((action, observation))
            
            # 5. 检查是否超过最大迭代次数
            if len(intermediate_steps) >= self.max_iterations:
                return self._handle_force_stop(intermediate_steps)
```

**关键数据结构：**

```python
from langchain_core.agents import AgentAction, AgentFinish

# Agent 决定调用工具时返回
AgentAction(
    tool="calculator",          # 工具名称
    tool_input={"expression": "2+2"},  # 工具参数
    log="Thought: 需要计算 2+2\nAction: calculator\nAction Input: ..."
)

# Agent 决定结束时返回  
AgentFinish(
    return_values={"output": "答案是 4"},
    log="Final Answer: 答案是 4"
)
```

### 1.5 现代 ReAct：Tool Calling Agent

LangChain 推荐使用 **Tool Calling Agent**（基于 OpenAI Function Calling / Claude Tool Use），相比文本解析更可靠：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}今天晴，26°C"

# 2. 定义 Prompt（包含 agent_scratchpad 占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的 AI 助手，可以使用工具回答问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. 创建 Agent
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. 包装成 AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "北京今天天气怎么样？"})
```

---

## 二、实践练习

### 题目 1：构建具有错误恢复能力的 ReAct Agent

**背景：** 在生产环境中，工具调用可能失败（网络超时、API 限流等）。设计一个 ReAct Agent，当工具调用失败时能够自动重试或切换备用方案。

**要求：**
1. 实现至少 2 个工具：`primary_search`（模拟 70% 失败率）和 `fallback_search`（稳定可用）
2. Agent 能够在 `primary_search` 失败后自动使用 `fallback_search`
3. 记录每次工具调用的成功/失败状态
4. `AgentExecutor` 设置合理的 `max_iterations` 和 `handle_parsing_errors`

**评分要点：**
- 工具的错误处理与日志记录（30%）
- Agent 的降级策略是否合理（30%）
- `AgentExecutor` 配置是否完整（20%）
- 代码可读性与注释（20%）

---

### 题目 1 标准答案

```python
"""
题目1：具有错误恢复能力的 ReAct Agent
知识点：AgentExecutor 配置、工具错误处理、降级策略
"""
import random
import logging
from langchain_core.tools import tool, ToolException
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def primary_search(query: str) -> str:
    """主搜索引擎，搜索互联网上的最新信息。速度快但不稳定。
    
    Args:
        query: 搜索关键词
    """
    # 模拟 70% 失败率
    if random.random() < 0.7:
        logger.warning(f"primary_search 调用失败: query={query}")
        raise ToolException(f"主搜索服务不可用，请使用备用搜索。query={query}")
    
    logger.info(f"primary_search 成功: query={query}")
    return f"主搜索结果[{query}]: 找到 5 条相关内容..."


@tool  
def fallback_search(query: str) -> str:
    """备用搜索引擎，搜索互联网信息。速度较慢但稳定可靠。
    
    Args:
        query: 搜索关键词
    """
    logger.info(f"fallback_search 调用: query={query}")
    return f"备用搜索结果[{query}]: 找到 3 条相关内容（来自缓存）..."


@tool
def calculate(expression: str) -> str:
    """安全地计算简单数学表达式（仅支持数字和 +、-、*、/ 运算）。
    
    Args:
        expression: 数学表达式，如 '2 + 3 * 4'
    """
    import ast
    import operator as op
    
    # 安全的运算符白名单（无 eval）
    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.USub: op.neg,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](safe_eval(node.operand))
        msg = f"不支持的表达式: {ast.dump(node)}"
        raise ToolException(msg)
    
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = safe_eval(tree.body)
        return f"计算结果: {expression} = {result}"
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"计算失败: {e}") from e


def create_resilient_agent() -> AgentExecutor:
    """创建具有错误恢复能力的 ReAct Agent。"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [primary_search, fallback_search, calculate]
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个智能助手，可以使用工具完成任务。
            
工具使用策略：
1. 首先尝试使用 primary_search 搜索信息
2. 如果 primary_search 失败，立即切换到 fallback_search
3. 对于数学计算，使用 calculate 工具
4. 遇到工具错误时，在回复中说明使用了哪个备用方案

始终保持耐心，确保用户得到正确答案。"""
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,           # 防止无限循环
        max_execution_time=30.0,    # 最大执行时间（秒）
        handle_parsing_errors=True, # 解析错误时不崩溃
        return_intermediate_steps=True,  # 返回中间步骤用于调试
    )


def run_demo():
    executor = create_resilient_agent()
    
    result = executor.invoke({
        "input": "请搜索'langchain agent'并计算 15 * 8 + 32"
    })
    
    print("\n=== 最终答案 ===")
    print(result["output"])
    
    print("\n=== 中间步骤 ===")
    for i, (action, observation) in enumerate(result["intermediate_steps"]):
        print(f"步骤 {i+1}: 调用 {action.tool}({action.tool_input})")
        print(f"  结果: {observation[:100]}...")


if __name__ == "__main__":
    run_demo()
```

**解析：**
- `ToolException` 是 LangChain 推荐的工具错误类型，Agent 能感知并在 Prompt 中自动处理
- `handle_parsing_errors=True` 防止 LLM 输出格式错误导致整个 Agent 崩溃
- `return_intermediate_steps=True` 对调试和可观测性至关重要
- `max_iterations` + `max_execution_time` 双重保护，避免"死循环"

---

### 题目 2：分析 AgentExecutor 的执行轨迹

**背景：** 调试 Agent 时需要理解每一步的决策。

**要求：**
1. 实现一个自定义 Callback Handler，记录 Agent 的每次 Thought、Action 和 Observation
2. 计算 Agent 完成任务的总 Token 消耗
3. 输出格式化的执行报告（包含时间戳、步骤数、总耗时）

---

### 题目 2 标准答案

```python
"""
题目2：Agent 执行轨迹分析与可观测性
知识点：Callback System、Token 统计、执行报告
"""
import time
from datetime import datetime
from typing import Any, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class AgentTraceCallback(BaseCallbackHandler):
    """记录 Agent 完整执行轨迹的 Callback Handler。"""
    
    def __init__(self):
        self.steps: list[dict] = []
        self.start_time: float = 0.0
        self.total_tokens: int = 0
        self.llm_calls: int = 0
    
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent 开始执行")
        print(f"输入: {inputs.get('input', inputs)}")
        print(f"{'='*50}")
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        step = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
        }
        self.steps.append(step)
        print(f"\n[{step['timestamp']}] 🔧 Action: {action.tool}")
        print(f"  Input: {action.tool_input}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        if self.steps:
            self.steps[-1]["observation"] = output[:200]
        print(f"  Observation: {output[:100]}{'...' if len(output) > 100 else ''}")
    
    def on_tool_error(self, error: Union[Exception, str], **kwargs) -> None:
        print(f"  ❌ Tool Error: {error}")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.llm_calls += 1
        # 统计 token 使用量
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_tokens += usage.get("total_tokens", 0)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        elapsed = time.time() - self.start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Agent 完成")
        print(f"\n{'='*50}")
        print("📊 执行报告")
        print(f"{'='*50}")
        print(f"总步骤数: {len(self.steps)}")
        print(f"LLM 调用次数: {self.llm_calls}")
        print(f"总 Token 消耗: {self.total_tokens}")
        print(f"总耗时: {elapsed:.2f}s")
        print(f"{'='*50}")
        print("\n执行轨迹:")
        for i, step in enumerate(self.steps, 1):
            print(f"  {i}. [{step['timestamp']}] {step['tool']}({step['input']})")
            if "observation" in step:
                print(f"     → {step['observation'][:80]}...")


@tool
def search_news(topic: str) -> str:
    """搜索最新新闻。Args: topic: 新闻主题"""
    return f"关于'{topic}'的最新新闻: [模拟数据] 今日头条..."


@tool
def get_stock_price(symbol: str) -> str:
    """获取股票价格。Args: symbol: 股票代码如 AAPL"""
    prices = {"AAPL": 185.5, "GOOGL": 140.2, "MSFT": 415.8}
    price = prices.get(symbol.upper(), 100.0)
    return f"{symbol} 当前价格: ${price}"


def run_traced_agent():
    """运行带轨迹追踪的 Agent。"""
    tracer = AgentTraceCallback()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[tracer])
    tools = [search_news, get_stock_price]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个财经分析助手，使用工具获取市场数据并分析。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[tracer],
        return_intermediate_steps=True,
    )
    
    result = executor.invoke({
        "input": "搜索苹果公司最新新闻，并获取 AAPL 的股票价格，给出简要分析"
    })
    
    print(f"\n最终答案:\n{result['output']}")


if __name__ == "__main__":
    run_traced_agent()
```

**解析：**
- `BaseCallbackHandler` 是 LangChain 可观测性的核心接口，面试中常被问到
- `on_agent_action` / `on_tool_end` / `on_agent_finish` 覆盖 Agent 生命周期的关键节点
- Token 统计在生产环境中对成本控制至关重要
- 此模式可直接对接 LangSmith、Prometheus 等监控系统

---

## 三、知识总结

### ReAct 模式核心要点

| 要点 | 说明 |
|------|------|
| Thought | LLM 的推理过程，不直接输出给用户 |
| Action | 工具调用，包含工具名和参数 |
| Observation | 工具执行结果，作为上下文输入下一轮 |
| Final Answer | Agent 判断信息充足，终止循环 |
| `intermediate_steps` | 完整的 (Action, Observation) 历史列表 |

### 面试高频问题

1. **ReAct 和 Chain-of-Thought 的区别？**
   - CoT 只有"思考"，没有与外部工具交互；ReAct 能调用工具获取真实信息

2. **AgentExecutor 如何防止无限循环？**
   - `max_iterations`（最大迭代次数）和 `max_execution_time`（最大执行时间）

3. **工具调用失败时 Agent 怎么处理？**
   - `handle_parsing_errors=True`；工具抛出 `ToolException` 会被包装成 Observation
