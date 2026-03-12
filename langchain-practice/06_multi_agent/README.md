# 06 多 Agent 协作系统

## 一、理论教学

### 6.1 为什么需要多 Agent？

单个 Agent 的局限性：
- 工具太多时，LLM 选择工具的准确率下降
- 复杂任务需要不同专业知识，单一 LLM 难以同时胜任
- 无法并行处理相互独立的子任务

多 Agent 系统的优势：
- **专业化**：每个 Agent 专注特定领域，减少"选择困难"
- **并行化**：多个 Agent 同时工作，提升吞吐
- **可扩展**：增加新能力只需增加新 Agent，无需修改现有 Agent
- **容错性**：单个 Agent 失败不影响整个系统

### 6.2 多 Agent 架构模式

**模式一：Supervisor（监督者）模式**（最常见）

```
用户输入
    ↓
Supervisor Agent（任务规划与路由）
    ├──→ Worker Agent A（数据分析）
    ├──→ Worker Agent B（代码生成）
    └──→ Worker Agent C（文档写作）
         ↓
      聚合结果
         ↓
      最终输出
```

**模式二：Pipeline（流水线）模式**

```
用户输入 → Agent1（数据收集）→ Agent2（分析）→ Agent3（报告生成）→ 输出
```

**模式三：Debate（辩论）模式**

```
问题 → Agent A（正方）→ Agent B（反方）→ Judge（裁判）→ 最终答案
```

**模式四：Hierarchical（层级）模式**

```
Top Supervisor
├── Mid Supervisor A
│   ├── Worker A1
│   └── Worker A2
└── Mid Supervisor B
    ├── Worker B1
    └── Worker B2
```

### 6.3 LangGraph 简介（多 Agent 推荐框架）

> LangGraph 是 LangChain 团队开发的图形化工作流框架，是实现多 Agent 系统的推荐工具。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 定义共享状态（所有 Agent 都能读写）
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息列表，支持追加
    next_agent: str                           # 下一个执行的 Agent
    final_answer: str                         # 最终答案

# 构建图
workflow = StateGraph(AgentState)

# 添加节点（每个节点是一个 Agent 或函数）
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# 添加边（定义流向）
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],  # 根据状态决定下一步
    {
        "researcher": "researcher",
        "writer": "writer",
        "FINISH": END,
    }
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")

# 编译
app = workflow.compile()
```

### 6.4 不使用 LangGraph 的纯 LangChain 多 Agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 定义专职 Worker Agent
def create_worker_agent(name: str, expertise: str, tools: list) -> AgentExecutor:
    """创建专职 Worker Agent。"""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是{name}，专注于{expertise}。只做你专长的事，完成后返回结果。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=3)


# Supervisor 通过工具调用各 Worker
@tool
def call_researcher(task: str) -> str:
    """调用研究员 Agent 收集信息。Args: task: 研究任务描述"""
    result = researcher_executor.invoke({"input": task})
    return result["output"]


@tool
def call_writer(content_to_write: str) -> str:
    """调用写作 Agent 生成文档。Args: content_to_write: 要写入的内容要求"""
    result = writer_executor.invoke({"input": content_to_write})
    return result["output"]
```

---

## 二、实践练习

### 题目 1：实现 Supervisor-Worker 多 Agent 系统

**背景：** 构建一个"AI 研究报告生成系统"，包含：
- **Supervisor**：接收用户请求，规划任务，协调子 Agent
- **Researcher Agent**：负责信息收集和整理
- **Analyst Agent**：负责数据分析
- **Writer Agent**：负责报告写作和格式化

**要求：**
1. Supervisor 能根据任务类型分发给合适的 Worker
2. Worker 的输出自动汇总给 Supervisor
3. 实现任务状态追踪（待分配/进行中/已完成）

---

### 题目 1 标准答案

```python
"""
题目1：Supervisor-Worker 多 Agent 系统
知识点：Agent 协调、任务分发、状态管理
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool, ToolException
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ===== 共享任务状态 =====

@dataclass
class Task:
    """任务数据类。"""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending/in_progress/completed/failed
    result: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class TaskRegistry:
    """全局任务注册表（模拟共享状态）。"""
    
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._counter = 0
    
    def create_task(self, description: str) -> Task:
        self._counter += 1
        task = Task(id=f"task_{self._counter:03d}", description=description)
        self._tasks[task.id] = task
        return task
    
    def assign_task(self, task_id: str, agent: str) -> None:
        task = self._tasks[task_id]
        task.assigned_to = agent
        task.status = "in_progress"
    
    def complete_task(self, task_id: str, result: str) -> None:
        task = self._tasks[task_id]
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now().isoformat()
    
    def fail_task(self, task_id: str, reason: str) -> None:
        task = self._tasks[task_id]
        task.status = "failed"
        task.result = f"失败原因: {reason}"
    
    def get_summary(self) -> str:
        stats = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for task in self._tasks.values():
            stats[task.status] = stats.get(task.status, 0) + 1
        return json.dumps(stats, ensure_ascii=False)
    
    def get_all_results(self) -> list[dict]:
        return [
            {"id": t.id, "desc": t.description[:50], "status": t.status,
             "agent": t.assigned_to, "result": (t.result or "")[:100]}
            for t in self._tasks.values()
        ]


# 全局任务注册表
task_registry = TaskRegistry()


# ===== Worker Agents =====

def create_researcher() -> AgentExecutor:
    """创建研究员 Agent。"""
    
    @tool
    def search_topic(topic: str) -> str:
        """搜索指定主题的信息。Args: topic: 研究主题"""
        # 模拟搜索
        return f"关于'{topic}'的研究发现：[1] 最新趋势显示... [2] 主要挑战包括... [3] 领先企业有..."
    
    @tool
    def collect_statistics(domain: str) -> str:
        """收集特定领域的统计数据。Args: domain: 数据领域"""
        return f"{domain}领域数据：市场规模 500 亿元（2024），YoY增长 35%，主要玩家占比 TOP3=60%"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业研究员。职责：
1. 使用工具搜集信息
2. 整理成结构化的研究报告
3. 注明数据来源
回答简洁，重点突出。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [search_topic, collect_statistics]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=False)


def create_analyst() -> AgentExecutor:
    """创建分析师 Agent。"""
    
    @tool
    def analyze_data(data: str) -> str:
        """分析数据，找出规律和洞察。Args: data: 原始数据字符串"""
        return f"数据分析结果：[关键洞察1] 增长主要驱动力是... [关键洞察2] 风险点在于... [建议] 应重点关注..."
    
    @tool
    def compare_competitors(companies: str) -> str:
        """对比分析竞争对手。Args: companies: 公司名称，逗号分隔"""
        return f"竞争对比分析（{companies}）：优势差异分析、市场定位分析、SWOT 对比..."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是数据分析师。职责：
1. 分析研究员提供的数据
2. 发现关键规律和趋势
3. 提出数据支持的结论
保持客观，数据说话。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [analyze_data, compare_competitors]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=False)


def create_writer() -> AgentExecutor:
    """创建写作 Agent。"""
    
    @tool
    def format_report(content: str, format_type: str = "markdown") -> str:
        """将内容格式化为指定格式的报告。
        Args:
            content: 报告内容
            format_type: 格式类型：markdown/plain/executive_summary
        """
        if format_type == "executive_summary":
            return f"【执行摘要】\n核心观点：{content[:200]}...\n\n关键数据：...\n\n行动建议：..."
        return f"# 研究报告\n\n## 摘要\n{content[:100]}...\n\n## 详细分析\n{content}\n\n---\n*生成时间: {datetime.now().strftime('%Y-%m-%d')}*"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业报告写作师。职责：
1. 将研究和分析内容整合成清晰报告
2. 确保结构清晰：摘要→正文→结论→建议
3. 语言专业、简洁
4. 使用 format_report 工具输出最终格式"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    tools = [format_report]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=False)


# ===== Supervisor =====

def create_supervisor(
    researcher: AgentExecutor,
    analyst: AgentExecutor,
    writer: AgentExecutor,
) -> callable:
    """
    创建 Supervisor，协调 Worker Agents 完成复杂任务。
    
    Supervisor 通过工具调用各 Worker，并汇总结果。
    """
    
    # 将 Worker 包装为 Supervisor 的工具
    @tool
    def delegate_to_researcher(research_task: str) -> str:
        """委托研究员进行信息收集。
        Args: research_task: 具体的研究任务描述，要明确、具体
        """
        task = task_registry.create_task(research_task)
        task_registry.assign_task(task.id, "researcher")
        
        print(f"  [Supervisor] 分配任务 {task.id} 给研究员: {research_task[:50]}")
        
        try:
            result = researcher.invoke({"input": research_task})
            output = result["output"]
            task_registry.complete_task(task.id, output)
            return f"[研究员报告 - {task.id}]\n{output}"
        except Exception as e:
            task_registry.fail_task(task.id, str(e))
            raise ToolException(f"研究员任务失败: {e}") from e
    
    @tool
    def delegate_to_analyst(analysis_task: str) -> str:
        """委托分析师进行数据分析。
        Args: analysis_task: 分析任务，包含需要分析的数据或问题
        """
        task = task_registry.create_task(analysis_task)
        task_registry.assign_task(task.id, "analyst")
        
        print(f"  [Supervisor] 分配任务 {task.id} 给分析师: {analysis_task[:50]}")
        
        try:
            result = analyst.invoke({"input": analysis_task})
            output = result["output"]
            task_registry.complete_task(task.id, output)
            return f"[分析师报告 - {task.id}]\n{output}"
        except Exception as e:
            task_registry.fail_task(task.id, str(e))
            raise ToolException(f"分析师任务失败: {e}") from e
    
    @tool
    def delegate_to_writer(writing_task: str) -> str:
        """委托写作师生成最终报告。
        Args: writing_task: 写作要求，包含研究和分析的汇总内容
        """
        task = task_registry.create_task(writing_task)
        task_registry.assign_task(task.id, "writer")
        
        print(f"  [Supervisor] 分配任务 {task.id} 给写作师: {writing_task[:50]}")
        
        try:
            result = writer.invoke({"input": writing_task})
            output = result["output"]
            task_registry.complete_task(task.id, output)
            return f"[写作师报告 - {task.id}]\n{output}"
        except Exception as e:
            task_registry.fail_task(task.id, str(e))
            raise ToolException(f"写作师任务失败: {e}") from e
    
    @tool
    def get_task_status() -> str:
        """获取所有任务的当前状态汇总。"""
        return task_registry.get_summary()
    
    # Supervisor Prompt
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是任务协调 Supervisor。职责：
1. 分析用户需求，规划完成任务的步骤
2. 将子任务分配给合适的 Worker（研究员/分析师/写作师）
3. 汇总各 Worker 的成果，生成最终答案

工作流程建议：
- 信息收集 → 委托研究员
- 数据分析 → 委托分析师（基于研究员的结果）
- 报告生成 → 委托写作师（基于前两步的结果）

确保任务之间的依赖关系正确，不要并行执行有依赖的任务。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    tools = [delegate_to_researcher, delegate_to_analyst, delegate_to_writer, get_task_status]
    supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    supervisor_agent = create_tool_calling_agent(supervisor_llm, tools, supervisor_prompt)
    
    supervisor_executor = AgentExecutor(
        agent=supervisor_agent,
        tools=tools,
        max_iterations=8,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return supervisor_executor


def run_multi_agent_demo():
    """运行多 Agent 协作示例。"""
    # 创建 Worker Agents
    researcher = create_researcher()
    analyst = create_analyst()
    writer = create_writer()
    
    # 创建 Supervisor
    supervisor = create_supervisor(researcher, analyst, writer)
    
    # 执行复杂任务
    print("=== 多 Agent 系统启动 ===\n")
    
    result = supervisor.invoke({
        "input": "请为'中国 AI 大模型市场'撰写一份研究报告，包括市场规模、主要玩家、竞争格局分析和未来趋势"
    })
    
    print("\n=== 最终报告 ===")
    print(result["output"])
    
    print("\n=== 任务执行统计 ===")
    for task_info in task_registry.get_all_results():
        status_icon = "✅" if task_info["status"] == "completed" else "❌"
        print(f"{status_icon} [{task_info['id']}] {task_info['agent']}: {task_info['desc']}")


if __name__ == "__main__":
    run_multi_agent_demo()
```

---

## 三、知识总结

### 多 Agent 架构选型

| 场景 | 推荐模式 | 工具 |
|------|---------|------|
| 固定步骤工作流 | Pipeline | LCEL Chain |
| 动态任务分发 | Supervisor | LangGraph / AgentExecutor |
| 专家协商 | Debate | LangGraph |
| 复杂层级任务 | Hierarchical | LangGraph |

### 面试高频问题

1. **多 Agent 系统的最大挑战是什么？**
   - Agent 间通信协议设计、状态同步、错误传播处理、成本控制（多 LLM 调用）

2. **LangGraph 和纯 LangChain AgentExecutor 的区别？**
   - LangGraph 支持循环图、共享状态、人工介入（human-in-the-loop）、更细粒度的控制流

3. **如何防止多 Agent 系统中的"死循环"？**
   - 全局最大步骤限制、每个 Agent 独立超时、状态机跟踪任务进度
