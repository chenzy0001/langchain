# 05 LCEL 自定义 Agent（LangChain Expression Language）

## 一、理论教学

### 5.1 什么是 LCEL？

**LCEL（LangChain Expression Language）** 是 LangChain 在 v0.1 引入的声明式编程接口，使用 Python 的 `|` 操作符将组件链式组合。

```python
# 旧方式（命令式）
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
result = chain.run(input="hello")

# LCEL 方式（声明式）
chain = prompt | llm | parser
result = chain.invoke({"input": "hello"})
```

### 5.2 LCEL 的核心优势

| 特性 | 说明 |
|------|------|
| **流式输出** | `.stream()` 逐 Token 输出，降低 TTFT |
| **批处理** | `.batch()` 并发处理多个输入 |
| **异步** | `.ainvoke()` / `.astream()` 原生异步 |
| **并行执行** | `RunnableParallel` 并发运行多个分支 |
| **条件路由** | `RunnableBranch` 按条件选择不同分支 |
| **可观测性** | 内建 LangSmith 追踪支持 |
| **类型安全** | `get_input_schema()` / `get_output_schema()` |

### 5.3 核心 Runnable 组件

```python
from langchain_core.runnables import (
    RunnablePassthrough,  # 透传输入，不做任何修改
    RunnableParallel,     # 并行执行多个 Runnable
    RunnableLambda,       # 将普通函数包装成 Runnable
    RunnableBranch,       # 条件分支
    RunnableWithFallbacks, # 带降级的 Runnable
)

# 1. RunnablePassthrough - 透传，常用于保留原始输入
chain = RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough(),  # 原样传递 question
)

# 2. RunnableLambda - 包装普通函数
def uppercase(text: str) -> str:
    return text.upper()

chain = RunnableLambda(uppercase)
chain.invoke("hello")  # "HELLO"

# 3. RunnableBranch - 条件路由
from langchain_core.runnables import RunnableBranch

router = RunnableBranch(
    (lambda x: "技术" in x["topic"], tech_chain),
    (lambda x: "政策" in x["topic"], policy_chain),
    default_chain,  # 默认分支
)

# 4. RunnableWithFallbacks - 降级策略
primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatOpenAI(model="gpt-4o-mini")

resilient_chain = (primary_llm | parser).with_fallbacks(
    [fallback_llm | parser],
    exceptions_to_handle=(Exception,),
)
```

### 5.4 从头构建自定义 LCEL Agent

LCEL Agent 的本质：**将 Agent 推理过程表达为 Runnable Pipeline**。

```python
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

@tool
def get_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_time]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)  # 绑定工具 Schema

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。使用工具完成任务。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def parse_output(message: AIMessage):
    """解析 LLM 输出为 AgentAction 或 AgentFinish。"""
    if message.tool_calls:
        return AgentAction(
            tool=message.tool_calls[0]["name"],
            tool_input=message.tool_calls[0]["args"],
            log=str(message),
        )
    return AgentFinish(return_values={"output": message.content}, log=str(message))

# LCEL Agent = Prompt → LLM（绑定工具）→ 解析输出
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
    }
    | prompt
    | llm
    | RunnableLambda(parse_output)
)

# 手动执行循环（理解 AgentExecutor 内部机制）
def run_agent(question: str) -> str:
    intermediate_steps = []
    
    for _ in range(5):  # 最多 5 轮
        output = agent.invoke({
            "input": question,
            "agent_scratchpad": [
                msg for action, obs in intermediate_steps
                for msg in [
                    AIMessage(content="", tool_calls=[{
                        "name": action.tool,
                        "args": action.tool_input,
                        "id": "call_0",
                    }]),
                ]
            ],
        })
        
        if isinstance(output, AgentFinish):
            return output.return_values["output"]
        
        # 执行工具
        tool_result = tools_by_name[output.tool].invoke(output.tool_input)
        intermediate_steps.append((output, tool_result))
    
    return "达到最大迭代次数"
```

### 5.5 高级 LCEL 模式

**模式一：动态路由 Agent**

```python
from langchain_core.output_parsers import StrOutputParser

def route_by_intent(chain_input: dict) -> Runnable:
    """根据意图动态选择不同的处理链。"""
    question = chain_input["question"]
    
    # 意图分类（简化版，实际可用 LLM 分类）
    if any(kw in question for kw in ["计算", "数学", "加减乘除"]):
        return math_chain
    elif any(kw in question for kw in ["搜索", "查找", "新闻"]):
        return search_chain
    else:
        return general_chain

dynamic_chain = RunnableLambda(route_by_intent)
```

**模式二：并行 Agent（扇出+聚合）**

```python
# 同时调用多个专业 Agent，聚合结果
parallel_agents = RunnableParallel(
    tech_analysis=tech_agent_chain,
    market_analysis=market_agent_chain,
    risk_assessment=risk_agent_chain,
) | RunnableLambda(lambda x: f"""
技术分析: {x['tech_analysis']}
市场分析: {x['market_analysis']}  
风险评估: {x['risk_assessment']}
""")
```

---

## 二、实践练习

### 题目 1：使用 LCEL 构建结构化输出 Agent

**背景：** 很多业务场景需要 Agent 输出严格的 JSON 格式（如生成报告、提取信息）。

**要求：**
1. 使用 LCEL 构建 Agent，输出结构化的 Pydantic 模型（包含 3+ 字段）
2. 实现 Schema 验证：当 LLM 输出不符合格式时自动重试（最多 3 次）
3. 使用 `with_structured_output()` 方法

---

### 题目 1 标准答案

```python
"""
题目1：结构化输出 LCEL Agent
知识点：with_structured_output、Pydantic 验证、重试机制
"""
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithFallbacks


class CompanyAnalysis(BaseModel):
    """公司分析报告结构。"""
    company_name: str = Field(description="公司名称")
    industry: str = Field(description="所属行业")
    strengths: list[str] = Field(
        description="公司核心优势，3-5条",
        min_length=2,
        max_length=5,
    )
    risks: list[str] = Field(
        description="主要风险因素，2-4条",
        min_length=1,
        max_length=4,
    )
    investment_rating: str = Field(
        description="投资评级：强烈推荐/推荐/中性/谨慎/回避",
    )
    target_price: Optional[float] = Field(
        default=None,
        description="目标股价（美元），如果无法判断则为 null",
    )
    summary: str = Field(description="一句话总结，50字以内")


def create_analysis_agent() -> callable:
    """
    构建公司分析 Agent，返回结构化 CompanyAnalysis。
    """
    # 主模型
    primary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 使用 with_structured_output 强制输出符合 Pydantic Schema 的 JSON
    structured_llm = primary_llm.with_structured_output(
        CompanyAnalysis,
        method="function_calling",  # 使用 Function Calling（更可靠）
    )
    
    # 备用模型（主模型失败时降级）
    fallback_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        CompanyAnalysis,
        method="function_calling",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的股票分析师。
        
根据提供的公司信息生成分析报告。要求：
- 分析要客观、专业
- 优势和风险各举具体事例
- 投资评级基于整体判断"""),
        ("human", "请分析以下公司：{company_info}"),
    ])
    
    # LCEL Chain: Prompt → 带结构化输出的 LLM
    # with_fallbacks 保证主模型失败时不崩溃
    chain = prompt | structured_llm.with_fallbacks(
        [fallback_llm],
        exception_key="error",
    )
    
    def analyze(company_info: str) -> CompanyAnalysis:
        """分析公司并返回结构化报告。"""
        return chain.invoke({"company_info": company_info})
    
    return analyze


def demo_structured_output():
    """演示结构化输出 Agent。"""
    analyzer = create_analysis_agent()
    
    company_info = """
    公司：OpenAI
    业务：开发 GPT-4、DALL-E 等 AI 模型，提供 API 服务
    融资：估值约 1570 亿美元，主要投资方 Microsoft
    收入：2024年 ARR 约 34 亿美元，增速超 100%
    竞争：面临 Google Gemini、Anthropic Claude、Meta LLaMA 等竞争
    """
    
    result = analyzer(company_info)
    
    print(f"公司: {result.company_name}")
    print(f"行业: {result.industry}")
    print(f"投资评级: {result.investment_rating}")
    print(f"目标股价: {'N/A' if result.target_price is None else f'${result.target_price}'}")
    print(f"\n核心优势:")
    for s in result.strengths:
        print(f"  + {s}")
    print(f"\n主要风险:")
    for r in result.risks:
        print(f"  - {r}")
    print(f"\n总结: {result.summary}")
    
    # 验证输出是 Pydantic 对象
    assert isinstance(result, CompanyAnalysis)
    print(f"\n✅ 输出格式验证通过，JSON: {result.model_dump_json(indent=2)}")


if __name__ == "__main__":
    demo_structured_output()
```

---

### 题目 2：构建多步骤 LCEL Pipeline（含分支与并行）

**要求：**
1. 实现一个"智能客服路由"Pipeline：
   - 步骤1：意图识别（技术支持 / 账单问题 / 投诉 / 其他）
   - 步骤2：根据意图路由到不同处理链
   - 步骤3：各链并行提取关键信息和生成回复
2. 使用 `RunnableBranch` 实现路由
3. 整个 Pipeline 支持 `.stream()` 流式输出

---

### 题目 2 标准答案

```python
"""
题目2：智能客服路由 Pipeline
知识点：RunnableBranch、RunnableParallel、流式输出、LCEL 组合
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI


# ===== Step 1: 意图识别 =====

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """对用户消息进行意图分类。
只返回以下之一：技术支持/账单问题/投诉/其他
不要任何额外解释。"""),
    ("human", "{message}"),
])

intent_chain = intent_prompt | llm | StrOutputParser()


# ===== Step 2: 各分支处理链 =====

def make_service_chain(role: str, guidance: str, temperature: float = 0.3) -> callable:
    """工厂函数：创建特定角色的客服链。
    
    Args:
        role: 角色描述
        guidance: 处理指引
        temperature: LLM 采样温度，0 表示确定性最强（适合技术支持），1 接近随机
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是{role}。{guidance}"),
        ("human", "用户问题：{message}\n意图：{intent}"),
    ])
    return prompt | ChatOpenAI(model="gpt-4o-mini", temperature=temperature) | StrOutputParser()


tech_chain = make_service_chain(
    "技术支持专家",
    "提供清晰的技术解决方案，步骤编号，语言简洁。",
    temperature=0.0,  # 技术问题需要确定性答案
)

billing_chain = make_service_chain(
    "账单服务专员",
    "耐心解释费用详情，提供退款/争议处理选项。",
    temperature=0.2,
)

complaint_chain = make_service_chain(
    "客户关系经理",
    "表达真诚歉意，承认问题，给出明确的解决时间表。",
    temperature=0.4,  # 投诉回复需要更有温度，允许一些灵活性
)

general_chain = make_service_chain(
    "通用客服助手",
    "友好、专业地回答问题，如需要转专业部门请说明。",
    temperature=0.3,
)


# ===== Step 3: 路由与执行 =====

def route_to_handler(inputs: dict) -> callable:
    """根据意图返回对应的处理链。"""
    intent = inputs.get("intent", "其他")
    
    routing = {
        "技术支持": tech_chain,
        "账单问题": billing_chain,
        "投诉": complaint_chain,
    }
    
    return routing.get(intent, general_chain)


def build_customer_service_pipeline():
    """
    构建完整的智能客服 Pipeline。
    
    流程：用户消息 → 意图识别 → 路由 → 专业处理 → 回复
    """
    
    def process(message: str) -> dict:
        """处理用户消息，返回意图和回复。"""
        # Step 1: 识别意图
        intent = intent_chain.invoke({"message": message})
        
        # Step 2: 路由到对应处理链
        handler = route_to_handler({"intent": intent})
        
        # Step 3: 生成回复（并行：生成回复 + 提取关键词）
        parallel = RunnableParallel(
            response=handler,
            keywords=ChatPromptTemplate.from_messages([
                ("system", "从用户问题中提取3个关键词，逗号分隔，只返回关键词。"),
                ("human", "{message}"),
            ]) | llm | StrOutputParser(),
        )
        
        result = parallel.invoke({"message": message, "intent": intent})
        
        return {
            "intent": intent,
            "response": result["response"],
            "keywords": result["keywords"],
        }
    
    return process


def demo_streaming():
    """演示流式输出。"""
    stream_chain = intent_prompt | llm | StrOutputParser()
    
    message = "我的账单上出现了一笔我不认识的费用，请帮我查查"
    
    print("=== 流式识别意图 ===")
    for chunk in stream_chain.stream({"message": message}):
        print(chunk, end="", flush=True)
    print()
    
    print("\n=== 流式生成技术支持回复 ===")
    tech_stream = ChatPromptTemplate.from_messages([
        ("system", "你是技术支持专家，提供清晰的解决方案。"),
        ("human", "{message}"),
    ]) | llm | StrOutputParser()
    
    for chunk in tech_stream.stream({"message": "我的 API 请求总是返回 429 错误"}):
        print(chunk, end="", flush=True)
    print()


def demo_pipeline():
    """演示完整的客服路由 Pipeline。"""
    pipeline = build_customer_service_pipeline()
    
    test_cases = [
        "我的应用连接数据库总是超时，错误代码是 ETIMEDOUT",
        "我上个月被多收了 200 元，要求退款",
        "服务太差了，我等了 3 天问题还没解决！",
    ]
    
    for message in test_cases:
        print(f"\n{'='*60}")
        print(f"用户: {message}")
        result = pipeline(message)
        print(f"意图: {result['intent']}")
        print(f"关键词: {result['keywords']}")
        print(f"\n回复:\n{result['response']}")


if __name__ == "__main__":
    demo_streaming()
    demo_pipeline()
```

---

## 三、知识总结

### LCEL 操作符速查

```python
# 基础链式组合
chain = a | b | c           # a的输出作为b的输入，依此类推

# 并行执行
chain = RunnableParallel(x=a, y=b)  # 同时运行 a 和 b

# 条件分支
chain = RunnableBranch((condition, branch_a), default_b)

# 带降级
chain = primary.with_fallbacks([fallback])

# 配置注入
chain = base_chain.configurable_fields(temperature=...)

# 绑定参数
chain = llm.bind(temperature=0, max_tokens=100)
```

### 面试高频问题

1. **LCEL 和旧式 Chain 最大的区别？**
   - 声明式 vs 命令式、原生流式支持、统一的 `invoke/stream/batch/astream` 接口

2. **如何在 LCEL Chain 中注入运行时参数？**
   - `chain.with_config()`、`RunnablePassthrough.assign()`、`configurable_fields()`

3. **RunnableParallel 的实际使用场景？**
   - 同时检索多个知识库、并行运行多个 LLM、扇出-聚合模式
