# 02 工具设计与管理（Tool Use）

## 一、理论教学

### 2.1 什么是 Tool？

**Tool（工具）** 是 Agent 与外部世界交互的接口。每个 Tool 都是一个具有：
- **名称（name）**：LLM 用来识别和选择工具的标识
- **描述（description）**：告诉 LLM 这个工具做什么、何时使用（**至关重要！**）
- **参数 Schema（args_schema）**：定义输入参数的类型和验证规则
- **执行函数（func）**：实际的业务逻辑

### 2.2 工具设计的核心原则

| 原则 | 说明 | 反例 |
|------|------|------|
| **单一职责** | 每个工具只做一件事 | 一个工具同时搜索+计算+存储 |
| **描述精准** | 描述决定 LLM 何时选择此工具 | `def search(q): """搜索"""` |
| **参数最小化** | 只暴露必要参数给 LLM | 让 LLM 传 API Key |
| **错误可读** | 抛出 LLM 能理解的错误信息 | `raise Exception("Error 500")` |
| **幂等性** | 相同输入总产生相同输出 | 有副作用的随机操作 |

### 2.3 工具的三种定义方式

**方式一：`@tool` 装饰器（最简单）**

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘并返回结果。
    
    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b

# 检查 Schema
print(multiply.name)          # "multiply"
print(multiply.description)   # 自动从 docstring 提取
print(multiply.args)          # {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}
```

**方式二：继承 `BaseTool`（完全控制）**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class DatabaseQueryInput(BaseModel):
    """数据库查询工具的输入参数。"""
    table: str = Field(description="要查询的数据库表名")
    condition: str = Field(description="WHERE 条件，如 'age > 18'")
    limit: int = Field(default=10, description="返回记录数量上限", ge=1, le=100)

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = (
        "查询关系型数据库。适用于需要精确数据的场景，"
        "如查找用户信息、订单记录等结构化数据。"
        "不适合全文搜索，请使用 search_tool 替代。"
    )
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    # 可以注入依赖（如数据库连接）
    db_connection: Optional[Any] = None
    
    def _run(self, table: str, condition: str, limit: int = 10) -> str:
        """同步执行（必须实现）。"""
        # 实际实现中连接数据库
        return f"查询 {table} WHERE {condition} LIMIT {limit}：找到 3 条记录"
    
    async def _arun(self, table: str, condition: str, limit: int = 10) -> str:
        """异步执行（可选，用于高性能场景）。"""
        # asyncpg 或其他异步驱动
        return self._run(table, condition, limit)
```

**方式三：`StructuredTool.from_function`（中间方案）**

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SendEmailInput(BaseModel):
    to: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文")

def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。"""
    # 实际发送逻辑
    return f"邮件已发送至 {to}，主题：{subject}"

email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="发送电子邮件。仅在用户明确要求发送邮件时使用。",
    args_schema=SendEmailInput,
    return_direct=False,  # True 时工具结果直接作为最终答案
)
```

### 2.4 工具路由与动态工具选择

在大型 Agent 系统中，工具可能有数十个。**工具路由**是决定何时使用哪个工具的机制：

```python
# 策略1：向量相似度路由（用于大量工具）
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_tool_retriever(tools: list) -> callable:
    """构建基于语义相似度的工具检索器。"""
    tool_docs = [
        {"name": t.name, "description": t.description, "tool": t}
        for t in tools
    ]
    
    vectorstore = FAISS.from_texts(
        texts=[f"{d['name']}: {d['description']}" for d in tool_docs],
        embedding=OpenAIEmbeddings(),
        metadatas=tool_docs,
    )
    
    def retrieve_tools(query: str, k: int = 3) -> list:
        """根据用户查询检索最相关的 k 个工具。"""
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.metadata["tool"] for doc in docs]
    
    return retrieve_tools
```

### 2.5 工具的安全与权限控制

```python
from functools import wraps
from langchain_core.tools import tool, ToolException

def require_permission(permission: str):
    """工具权限装饰器。"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user_context: dict = None, **kwargs):
            if user_context is None or permission not in user_context.get("permissions", []):
                raise ToolException(
                    f"权限不足：需要 '{permission}' 权限才能执行此操作。"
                    f"请联系管理员授权。"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


@tool
@require_permission("file_write")
def write_file(path: str, content: str) -> str:
    """写入文件内容。需要 file_write 权限。
    
    Args:
        path: 文件路径（相对路径）
        content: 要写入的内容
    """
    # 安全检查：防止路径穿越
    if ".." in path or path.startswith("/"):
        raise ToolException(f"非法路径：{path}。只允许相对路径。")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件 {path} 写入成功，共 {len(content)} 字符"
```

---

## 二、实践练习

### 题目 1：设计一个结构化的数据分析工具集

**背景：** 为数据分析 Agent 设计一套工具，支持数据查询、统计分析和可视化描述。

**要求：**
1. 使用 Pydantic 定义严格的输入 Schema，包含参数验证
2. 实现 3 个工具：`query_data`（查询数据）、`compute_statistics`（计算统计量）、`describe_chart`（描述图表数据，不实际绘图）
3. 工具描述要清晰区分使用场景，让 LLM 能正确选择
4. 每个工具需有适当的错误处理

---

### 题目 1 标准答案

```python
"""
题目1：数据分析工具集设计
知识点：BaseTool、Pydantic Schema、工具描述最佳实践
"""
from typing import Literal, Optional
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, field_validator
import statistics
import json


# ===== 工具 1：数据查询 =====

class QueryDataInput(BaseModel):
    """数据查询工具输入。"""
    dataset: Literal["sales", "users", "products"] = Field(
        description="要查询的数据集名称，可选: sales(销售), users(用户), products(产品)"
    )
    filters: Optional[dict] = Field(
        default=None,
        description="过滤条件，如 {'region': '华南', 'year': 2024}"
    )
    columns: Optional[list[str]] = Field(
        default=None,
        description="要返回的列名列表，None 表示返回所有列"
    )
    limit: int = Field(
        default=5,
        description="返回记录数量",
        ge=1,
        le=1000
    )

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Optional[dict]) -> Optional[dict]:
        """验证过滤条件不包含危险字段。"""
        if v and any(k.startswith("__") for k in v):
            msg = "过滤条件不能包含以 '__' 开头的字段"
            raise ValueError(msg)
        return v


class QueryDataTool(BaseTool):
    """数据查询工具。"""
    
    name: str = "query_data"
    description: str = (
        "从指定数据集中查询原始数据记录。"
        "适用于：获取具体记录、查看数据样本、按条件过滤数据。"
        "不适用于：计算统计量（请用 compute_statistics）、生成图表描述（请用 describe_chart）。"
    )
    args_schema: type[BaseModel] = QueryDataInput
    
    # 模拟数据库
    _mock_data: dict = {
        "sales": [
            {"id": 1, "product": "A", "amount": 1500, "region": "华南", "year": 2024},
            {"id": 2, "product": "B", "amount": 2300, "region": "华北", "year": 2024},
            {"id": 3, "product": "A", "amount": 1800, "region": "华东", "year": 2024},
            {"id": 4, "product": "C", "amount": 900, "region": "华南", "year": 2023},
            {"id": 5, "product": "B", "amount": 3100, "region": "华南", "year": 2024},
        ],
        "users": [
            {"id": 1, "name": "张三", "age": 28, "city": "上海", "vip": True},
            {"id": 2, "name": "李四", "age": 35, "city": "北京", "vip": False},
        ],
        "products": [
            {"id": "A", "name": "产品A", "price": 299, "category": "电子"},
            {"id": "B", "name": "产品B", "price": 599, "category": "服装"},
        ],
    }
    
    def _run(
        self,
        dataset: str,
        filters: Optional[dict] = None,
        columns: Optional[list[str]] = None,
        limit: int = 5,
    ) -> str:
        data = self._mock_data.get(dataset, [])
        
        # 应用过滤
        if filters:
            data = [
                row for row in data
                if all(str(row.get(k)) == str(v) for k, v in filters.items())
            ]
        
        # 选择列
        if columns:
            data = [{k: row[k] for k in columns if k in row} for row in data]
        
        # 限制数量
        data = data[:limit]
        
        if not data:
            return f"数据集 '{dataset}' 在给定条件下没有找到记录。"
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ===== 工具 2：统计计算 =====

class ComputeStatisticsInput(BaseModel):
    """统计计算工具输入。"""
    data: list[float] = Field(
        description="要计算统计量的数值列表",
        min_length=1
    )
    metrics: list[Literal["mean", "median", "std", "min", "max", "sum", "count"]] = Field(
        default=["mean", "median", "std", "min", "max"],
        description="要计算的统计指标列表"
    )


class ComputeStatisticsTool(BaseTool):
    """统计计算工具。"""
    
    name: str = "compute_statistics"
    description: str = (
        "对数值列表计算统计指标（均值、中位数、标准差等）。"
        "适用于：数据汇总分析、分布特征描述、异常值检测。"
        "需要先用 query_data 获取数据，提取数值列后再传入此工具。"
    )
    args_schema: type[BaseModel] = ComputeStatisticsInput
    
    def _run(
        self,
        data: list[float],
        metrics: list[str] = None,
    ) -> str:
        if metrics is None:
            metrics = ["mean", "median", "std", "min", "max"]
        
        if len(data) == 0:
            raise ToolException("数据列表不能为空")
        
        calculators = {
            "mean": lambda d: round(statistics.mean(d), 2),
            "median": lambda d: round(statistics.median(d), 2),
            "std": lambda d: round(statistics.stdev(d), 2) if len(d) > 1 else 0,
            "min": min,
            "max": max,
            "sum": sum,
            "count": len,
        }
        
        results = {}
        for metric in metrics:
            if metric not in calculators:
                raise ToolException(f"不支持的统计指标: {metric}")
            results[metric] = calculators[metric](data)
        
        return json.dumps(results, ensure_ascii=False)
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ===== 工具 3：图表描述 =====

class DescribeChartInput(BaseModel):
    """图表描述工具输入。"""
    chart_type: Literal["bar", "line", "pie", "scatter"] = Field(
        description="图表类型：bar(柱状图), line(折线图), pie(饼图), scatter(散点图)"
    )
    x_data: list[str] = Field(description="X 轴数据标签列表")
    y_data: list[float] = Field(description="Y 轴数值列表，长度需与 x_data 一致")
    title: str = Field(description="图表标题")
    
    @field_validator("y_data")
    @classmethod
    def validate_y_data_length(cls, v: list, info) -> list:
        """验证 x 和 y 数据长度一致。"""
        # 注意: 在 Pydantic v2 中通过 model_validator 更优，此处简化
        return v


class DescribeChartTool(BaseTool):
    """图表数据描述工具（生成文字描述，不实际绘图）。"""
    
    name: str = "describe_chart"
    description: str = (
        "根据数据生成图表的文字描述，包括关键洞察和趋势分析。"
        "适用于：生成数据报告、描述可视化内容、提取关键规律。"
        "注意：此工具不实际生成图片，只生成文字描述。"
    )
    args_schema: type[BaseModel] = DescribeChartInput
    
    def _run(
        self,
        chart_type: str,
        x_data: list[str],
        y_data: list[float],
        title: str,
    ) -> str:
        if len(x_data) != len(y_data):
            raise ToolException(
                f"x_data ({len(x_data)} 项) 和 y_data ({len(y_data)} 项) 长度不一致"
            )
        
        max_idx = y_data.index(max(y_data))
        min_idx = y_data.index(min(y_data))
        avg = round(sum(y_data) / len(y_data), 2)
        
        chart_types_cn = {
            "bar": "柱状图", "line": "折线图",
            "pie": "饼图", "scatter": "散点图"
        }
        
        description = (
            f"【{title}】{chart_types_cn.get(chart_type, chart_type)}\n"
            f"数据点数量: {len(x_data)}\n"
            f"最高值: {x_data[max_idx]} = {y_data[max_idx]}\n"
            f"最低值: {x_data[min_idx]} = {y_data[min_idx]}\n"
            f"平均值: {avg}\n"
            f"总计: {sum(y_data)}"
        )
        
        if chart_type == "line" and len(y_data) > 1:
            trend = "上升" if y_data[-1] > y_data[0] else "下降"
            description += f"\n趋势: {trend}（从 {y_data[0]} 到 {y_data[-1]}）"
        
        return description
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ===== 组合使用示例 =====

def create_data_analyst_agent():
    """创建数据分析 Agent。"""
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    tools = [QueryDataTool(), ComputeStatisticsTool(), DescribeChartTool()]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业数据分析师。分析流程：
1. 用 query_data 获取原始数据
2. 提取数值列，用 compute_statistics 计算统计量
3. 用 describe_chart 生成可视化描述
4. 综合给出业务洞察"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    agent = create_data_analyst_agent()
    result = agent.invoke({
        "input": "分析华南地区2024年的销售数据，给出统计摘要和趋势描述"
    })
    print(result["output"])
```

---

### 题目 2：实现工具调用结果的缓存与去重

**要求：**
1. 为任意工具添加缓存装饰器，相同输入直接返回缓存结果
2. 缓存支持 TTL（过期时间）
3. 统计缓存命中率

---

### 题目 2 标准答案

```python
"""
题目2：工具结果缓存装饰器
知识点：工具扩展、缓存策略、性能监控
"""
import hashlib
import json
import time
from functools import wraps
from typing import Any, Optional
from langchain_core.tools import tool, ToolException


class ToolCache:
    """工具调用结果缓存，支持 TTL。"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Args:
            default_ttl: 默认缓存过期时间（秒），默认 5 分钟
        """
        self._cache: dict[str, dict] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, tool_name: str, kwargs: dict) -> str:
        """生成缓存 Key（工具名 + 参数的哈希值）。"""
        payload = json.dumps({"tool": tool_name, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]
    
    def get(self, tool_name: str, kwargs: dict) -> Optional[Any]:
        """获取缓存值，未命中或过期返回 None。"""
        key = self._make_key(tool_name, kwargs)
        entry = self._cache.get(key)
        
        if entry is None:
            self.misses += 1
            return None
        
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return entry["value"]
    
    def set(self, tool_name: str, kwargs: dict, value: Any, ttl: Optional[int] = None) -> None:
        """存储缓存值。"""
        key = self._make_key(tool_name, kwargs)
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + (ttl or self.default_ttl),
            "created_at": time.time(),
        }
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率。"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> dict:
        """返回缓存统计信息。"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "cached_items": len(self._cache),
        }


# 全局缓存实例（生产中可用 Redis 替代）
_global_cache = ToolCache(default_ttl=300)


def cached_tool(ttl: int = 300, cache: Optional[ToolCache] = None):
    """
    工具缓存装饰器。
    
    Args:
        ttl: 缓存过期时间（秒）
        cache: 自定义缓存实例，None 时使用全局缓存
    
    Usage:
        @cached_tool(ttl=60)
        @tool
        def search_web(query: str) -> str:
            ...
    """
    _cache = cache or _global_cache
    
    def decorator(tool_func):
        @wraps(tool_func)
        def wrapper(**kwargs):
            tool_name = tool_func.name
            
            # 尝试读取缓存
            cached_result = _cache.get(tool_name, kwargs)
            if cached_result is not None:
                print(f"[Cache HIT] {tool_name}({kwargs})")
                return cached_result
            
            # 缓存未命中，执行工具
            print(f"[Cache MISS] {tool_name}({kwargs})")
            result = tool_func.invoke(kwargs)
            
            # 存储到缓存（异常结果不缓存）
            if not isinstance(result, Exception):
                _cache.set(tool_name, kwargs, result, ttl=ttl)
            
            return result
        
        # 保留原工具的元数据
        wrapper.name = tool_func.name
        wrapper.description = tool_func.description
        wrapper.args = tool_func.args
        return wrapper
    
    return decorator


# ===== 使用示例 =====

@tool
def fetch_exchange_rate(from_currency: str, to_currency: str) -> str:
    """获取两种货币之间的实时汇率。
    
    Args:
        from_currency: 源货币代码，如 USD
        to_currency: 目标货币代码，如 CNY
    """
    import random
    time.sleep(0.1)  # 模拟 API 延迟
    rate = random.uniform(6.9, 7.3) if from_currency == "USD" else random.uniform(0.1, 0.5)
    return f"1 {from_currency} = {rate:.4f} {to_currency}（更新时间: {time.strftime('%H:%M:%S')}）"


# 对工具应用缓存
cached_exchange = cached_tool(ttl=60)(_global_cache)


def demo_cache():
    """演示缓存效果。"""
    print("=== 第一次调用（缓存未命中）===")
    start = time.time()
    r1 = fetch_exchange_rate.invoke({"from_currency": "USD", "to_currency": "CNY"})
    print(f"结果: {r1}，耗时: {(time.time()-start)*1000:.1f}ms")
    
    # 手动存入缓存演示
    _global_cache.set("fetch_exchange_rate", 
                       {"from_currency": "USD", "to_currency": "CNY"}, 
                       r1, ttl=60)
    
    print("\n=== 第二次调用（从缓存读取）===")
    start = time.time()
    r2 = _global_cache.get("fetch_exchange_rate", 
                            {"from_currency": "USD", "to_currency": "CNY"})
    print(f"结果: {r2}，耗时: {(time.time()-start)*1000:.1f}ms")
    
    print(f"\n=== 缓存统计 ===")
    print(json.dumps(_global_cache.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_cache()
```

---

## 三、知识总结

### 工具设计检查清单

```
✅ 名称清晰，动词+名词格式（如 search_web, calculate_tax）
✅ 描述包含：用途、适用场景、不适用场景
✅ Pydantic Schema 含参数验证（类型、范围、枚举）
✅ 错误使用 ToolException（让 Agent 能感知）
✅ 实现 _arun（支持异步 Agent）
✅ 工具是幂等的（相同输入，相同输出）
✅ 敏感参数不暴露给 LLM（如 API Key）
```

### 面试高频问题

1. **工具描述对 Agent 行为有什么影响？**
   - 描述直接影响 LLM 的工具选择决策，描述越精准，工具使用越准确

2. **`return_direct=True` 是什么意思？**
   - 工具执行结果直接作为 Agent 的最终答案，跳过后续 LLM 处理

3. **如何处理需要认证的工具？**
   - 在工具实例化时注入凭证（依赖注入），不让 LLM 参与认证流程
