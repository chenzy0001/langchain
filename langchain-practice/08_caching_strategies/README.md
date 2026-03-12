# 08 缓存策略（Caching Strategies）

## 一、理论教学

### 8.1 为什么需要缓存？

LLM API 调用的成本构成：

```
总成本 = 请求数 × 单次成本
单次成本 = (输入 Token 数 × 输入单价) + (输出 Token 数 × 输出单价)

以 gpt-4o-mini 为例：
- 输入: $0.150 / 1M tokens
- 输出: $0.600 / 1M tokens

一个 RAG 应用（每请求约 2000 tokens）：
- 1000 次/天 × $0.002/次 = $2/天 = $60/月
- 如果命中率 60%，成本降至 $24/月，节省 60%
```

### 8.2 LangChain 缓存层级

```
┌─────────────────────────────────────────┐
│  用户请求                                 │
│     ↓                                   │
│  ┌──────────────────────────────────┐   │
│  │  Layer 1: LLM 语义缓存           │   │
│  │  (完全相同/语义相似的查询命中)    │   │
│  └──────────┬───────────────────────┘   │
│             │ 未命中                     │
│  ┌──────────▼───────────────────────┐   │
│  │  Layer 2: Embedding 缓存         │   │
│  │  (向量化结果复用)                 │   │
│  └──────────┬───────────────────────┘   │
│             │ 未命中                     │
│  ┌──────────▼───────────────────────┐   │
│  │  Layer 3: 检索结果缓存           │   │
│  │  (相同 Query 的 VectorStore 查询) │   │
│  └──────────┬───────────────────────┘   │
│             ↓                           │
│         LLM API 调用                    │
└─────────────────────────────────────────┘
```

### 8.3 Layer 1：LLM 响应缓存

**内存缓存（开发环境）：**

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# 设置全局缓存（影响所有 LLM 调用）
set_llm_cache(InMemoryCache())

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# 第一次调用：API 请求（~500ms）
response1 = llm.invoke("Python 是什么？")

# 第二次调用：缓存命中（<1ms）
response2 = llm.invoke("Python 是什么？")  # 完全相同 → 直接返回缓存
```

**Redis 缓存（生产环境）：**

```python
from langchain_community.cache import RedisCache
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)

set_llm_cache(RedisCache(
    redis_=redis_client,
    ttl=3600,  # 缓存 1 小时
))
```

**语义缓存（相似查询复用）：**

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

# 语义缓存：不需要完全相同，相似语义也能命中
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95,  # 相似度阈值（0-1，越高越严格）
))

# "Python 是什么？" 和 "介绍一下 Python" 可能命中同一缓存
```

**SQLite 缓存（简单持久化）：**

```python
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
# 跨进程持久化，重启不丢失
```

### 8.4 Layer 2：Embedding 缓存

向量化是 RAG 中计算密集且可缓存的操作：

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore, InMemoryByteStore
from langchain_openai import OpenAIEmbeddings

# 底层 Embedding 模型
base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 本地文件缓存（持久化）
file_store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=file_store,
    namespace=base_embeddings.model,  # 命名空间避免不同模型冲突
)

# 使用方式完全相同，第一次 API 调用，之后从缓存读取
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(["some text"], cached_embeddings)
```

### 8.5 Layer 3：自定义缓存装饰器

```python
from functools import wraps
import hashlib, json, time
from typing import Any, Optional

class TieredCache:
    """多级缓存：L1 内存（快）→ L2 Redis（持久）。"""
    
    def __init__(self, redis_client=None, l1_max_size=100, l2_ttl=3600):
        self._l1: dict = {}         # 内存缓存（进程级）
        self._l1_max = l1_max_size
        self._l2 = redis_client      # Redis（跨进程）
        self._l2_ttl = l2_ttl
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Any]:
        # L1 查找
        if key in self._l1:
            self.stats["l1_hits"] += 1
            return self._l1[key]
        
        # L2 查找
        if self._l2:
            value = self._l2.get(f"cache:{key}")
            if value:
                data = json.loads(value)
                self._l1[key] = data  # 回填 L1
                self.stats["l2_hits"] += 1
                return data
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        # L1：LRU 简单实现（超出限制时删除最旧的）
        if len(self._l1) >= self._l1_max:
            oldest = next(iter(self._l1))
            del self._l1[oldest]
        self._l1[key] = value
        
        # L2
        if self._l2:
            self._l2.setex(f"cache:{key}", self._l2_ttl, json.dumps(value))
```

---

## 二、实践练习

### 题目 1：实现带命中率统计的多级 LLM 缓存系统

**要求：**
1. 实现 `SmartLLMCache` 类，继承 `BaseCache`
2. 支持精确匹配缓存（完全相同的 Prompt）
3. 实现 `get_stats()` 返回缓存命中率、节省的 API 调用次数、预估节省成本
4. 支持手动 `invalidate()` 使特定缓存失效

---

### 题目 1 标准答案

```python
"""
题目1：智能 LLM 缓存系统
知识点：BaseCache 继承、命中率统计、成本计算
"""
import hashlib
import json
import time
from typing import Any, Optional, Union
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation, ChatGeneration
from langchain_core.globals import set_llm_cache


class SmartLLMCache(BaseCache):
    """
    带统计功能的智能 LLM 缓存。
    
    特性：
    - 精确匹配缓存
    - 命中率统计
    - 成本节省估算
    - 手动失效
    - TTL 过期
    """
    
    # GPT-4o-mini 定价（美元/百万 Token）
    COST_PER_M_INPUT_TOKENS = 0.15
    COST_PER_M_OUTPUT_TOKENS = 0.60
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: 缓存过期时间（秒），默认 1 小时
        """
        self._cache: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._total_input_tokens_saved = 0
        self._total_output_tokens_saved = 0
    
    def _make_key(self, prompt: str, llm_string: str) -> str:
        """生成缓存 Key（Prompt + LLM 配置的哈希）。"""
        payload = f"{prompt}::{llm_string}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    def lookup(
        self,
        prompt: str,
        llm_string: str,
    ) -> Optional[list[Generation]]:
        """查找缓存（LangChain 框架调用）。"""
        key = self._make_key(prompt, llm_string)
        entry = self._cache.get(key)
        
        if entry is None:
            self._misses += 1
            return None
        
        # 检查 TTL
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            self._misses += 1
            return None
        
        self._hits += 1
        
        # 统计节省的 Token（估算）
        output_text = entry["generations"][0].text if entry["generations"] else ""
        # 粗略估算：4字符≈1Token
        self._total_input_tokens_saved += len(prompt) // 4
        self._total_output_tokens_saved += len(output_text) // 4
        
        print(f"[Cache HIT] key={key[:8]}... 已节省 {len(prompt)//4} 输入 Token")
        
        return entry["generations"]
    
    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: list[Generation],
    ) -> None:
        """更新缓存（LangChain 框架调用）。"""
        key = self._make_key(prompt, llm_string)
        self._cache[key] = {
            "generations": return_val,
            "expires_at": time.time() + self.ttl,
            "prompt_preview": prompt[:50],
        }
        print(f"[Cache SET] key={key[:8]}...")
    
    async def alookup(
        self,
        prompt: str,
        llm_string: str,
    ) -> Optional[list[Generation]]:
        """异步查找（直接调用同步版本）。"""
        return self.lookup(prompt, llm_string)
    
    async def aupdate(
        self,
        prompt: str,
        llm_string: str,
        return_val: list[Generation],
    ) -> None:
        """异步更新（直接调用同步版本）。"""
        self.update(prompt, llm_string, return_val)
    
    def invalidate(self, prompt: str, llm_string: str) -> bool:
        """手动使特定缓存失效。
        
        Returns:
            bool: True 表示成功删除，False 表示缓存不存在
        """
        key = self._make_key(prompt, llm_string)
        if key in self._cache:
            del self._cache[key]
            print(f"[Cache INVALIDATE] key={key[:8]}... 已删除")
            return True
        return False
    
    def clear(self, **kwargs: Any) -> None:
        """清空所有缓存。"""
        count = len(self._cache)
        self._cache.clear()
        print(f"[Cache CLEAR] 已清除 {count} 条缓存")
    
    def get_stats(self) -> dict:
        """
        返回缓存统计信息，包含节省成本估算。
        
        Returns:
            dict: 统计信息
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        # 成本节省估算
        input_cost_saved = (self._total_input_tokens_saved / 1_000_000) * self.COST_PER_M_INPUT_TOKENS
        output_cost_saved = (self._total_output_tokens_saved / 1_000_000) * self.COST_PER_M_OUTPUT_TOKENS
        total_cost_saved = input_cost_saved + output_cost_saved
        
        return {
            "total_requests": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_items": len(self._cache),
            "tokens_saved": {
                "input": self._total_input_tokens_saved,
                "output": self._total_output_tokens_saved,
            },
            "cost_saved_usd": round(total_cost_saved, 6),
            "cost_saved_cny": round(total_cost_saved * 7.2, 4),
        }


def demo_smart_cache():
    """演示智能缓存效果。"""
    import time
    
    # 安装缓存
    cache = SmartLLMCache(ttl_seconds=300)
    set_llm_cache(cache)
    
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    questions = [
        "什么是机器学习？",
        "解释深度学习",
        "什么是机器学习？",    # 重复 → 缓存命中
        "机器学习是什么？",    # 不同措辞 → 不命中（需语义缓存）
        "解释深度学习",        # 重复 → 缓存命中
    ]
    
    timings = []
    
    for q in questions:
        start = time.time()
        response = llm.invoke(q)
        elapsed = (time.time() - start) * 1000
        timings.append(elapsed)
        print(f"[{elapsed:.1f}ms] Q: {q[:20]}... → A: {response.content[:30]}...")
    
    print("\n=== 缓存统计 ===")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== 手动失效测试 ===")
    invalidated = cache.invalidate("什么是机器学习？", llm._llm_type)
    print(f"失效结果: {invalidated}")
    
    # 再次查询，应该缓存未命中
    start = time.time()
    llm.invoke("什么是机器学习？")
    elapsed = (time.time() - start) * 1000
    print(f"失效后重新查询耗时: {elapsed:.1f}ms（应显著高于缓存命中时间）")


if __name__ == "__main__":
    demo_smart_cache()
```

---

### 题目 2：实现 RAG 全链路缓存优化

**要求：**
1. 对 RAG Pipeline 的每个环节分别实施缓存策略
2. 实现 `CachedRAGPipeline` 类，包含 Embedding 缓存 + LLM 缓存 + 检索结果缓存
3. 提供 `benchmark()` 方法，对比有/无缓存的性能差异

---

### 题目 2 标准答案

```python
"""
题目2：RAG 全链路缓存优化
知识点：CacheBackedEmbeddings、多级缓存、性能基准测试
"""
import time
import hashlib
from typing import Optional
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


SAMPLE_DOCS = [
    Document(page_content="LangChain 是用于构建 LLM 应用的开源框架", metadata={"id": "1"}),
    Document(page_content="RAG 通过检索外部知识增强 LLM 的回答准确性", metadata={"id": "2"}),
    Document(page_content="向量数据库存储文本的语义向量表示", metadata={"id": "3"}),
    Document(page_content="Agent 通过工具调用与外部系统交互", metadata={"id": "4"}),
]


class CachedRAGPipeline:
    """
    带全链路缓存的 RAG Pipeline。
    
    三级缓存：
    1. Embedding 缓存（避免重复向量化）
    2. 检索结果缓存（相同查询不重复检索）
    3. LLM 响应缓存（相同问题+上下文不重复调用 LLM）
    """
    
    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self._retrieval_cache: dict[str, list[Document]] = {}
        self._retrieval_hits = 0
        self._retrieval_misses = 0
        
        # 配置 Embedding（带缓存或不带缓存）
        base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if enable_cache:
            byte_store = InMemoryByteStore()
            self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings=base_embeddings,
                document_embedding_cache=byte_store,
                namespace="rag_cache",
            )
            set_llm_cache(InMemoryCache())
        else:
            self._embeddings = base_embeddings
            set_llm_cache(None)
        
        # 构建向量库
        self._vectorstore = FAISS.from_documents(SAMPLE_DOCS, self._embeddings)
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 构建 LLM Chain
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "基于以下上下文回答问题，保持简洁：\n{context}"),
            ("human", "{question}"),
        ])
        self._chain = self._prompt | self._llm | StrOutputParser()
    
    def _cached_retrieve(self, query: str) -> list[Document]:
        """带缓存的检索。"""
        if not self.enable_cache:
            self._retrieval_misses += 1
            return self._retriever.invoke(query)
        
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:32]
        
        if cache_key in self._retrieval_cache:
            self._retrieval_hits += 1
            return self._retrieval_cache[cache_key]
        
        self._retrieval_misses += 1
        docs = self._retriever.invoke(query)
        self._retrieval_cache[cache_key] = docs
        return docs
    
    def answer(self, question: str) -> dict:
        """回答问题并返回结果和耗时。"""
        start = time.time()
        
        docs = self._cached_retrieve(question)
        context = "\n".join(d.page_content for d in docs)
        
        response = self._chain.invoke({"question": question, "context": context})
        
        latency_ms = round((time.time() - start) * 1000, 1)
        
        return {
            "answer": response,
            "latency_ms": latency_ms,
            "docs_count": len(docs),
        }
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计。"""
        total = self._retrieval_hits + self._retrieval_misses
        return {
            "retrieval_hits": self._retrieval_hits,
            "retrieval_misses": self._retrieval_misses,
            "retrieval_hit_rate": f"{self._retrieval_hits/total:.1%}" if total else "N/A",
        }
    
    def benchmark(self, questions: list[str], repeat: int = 2) -> dict:
        """基准测试：有/无缓存的性能对比。"""
        all_questions = questions * repeat  # 重复查询，模拟缓存效果
        
        timings = []
        for q in all_questions:
            result = self.answer(q)
            timings.append(result["latency_ms"])
        
        first_run = timings[:len(questions)]
        cache_run = timings[len(questions):]
        
        return {
            "cache_enabled": self.enable_cache,
            "questions_count": len(questions),
            "first_run_avg_ms": round(sum(first_run) / len(first_run), 1),
            "cached_run_avg_ms": round(sum(cache_run) / len(cache_run), 1),
            "speedup": round(sum(first_run) / sum(cache_run), 2) if self.enable_cache else 1.0,
            "cache_stats": self.get_cache_stats(),
        }


def compare_cached_vs_uncached():
    """对比有/无缓存的性能差异。"""
    questions = [
        "什么是 LangChain？",
        "RAG 有什么作用？",
        "向量数据库是什么？",
    ]
    
    print("=== 测试无缓存 RAG ===")
    uncached = CachedRAGPipeline(enable_cache=False)
    result_no_cache = uncached.benchmark(questions)
    
    print("\n=== 测试带缓存 RAG ===")
    cached = CachedRAGPipeline(enable_cache=True)
    result_with_cache = cached.benchmark(questions)
    
    print("\n=== 性能对比报告 ===")
    print(f"{'指标':<25} {'无缓存':>15} {'有缓存':>15}")
    print("-" * 55)
    print(f"{'首次查询平均延迟':<25} {result_no_cache['first_run_avg_ms']:>14}ms "
          f"{result_with_cache['first_run_avg_ms']:>14}ms")
    print(f"{'重复查询平均延迟':<25} {result_no_cache['cached_run_avg_ms']:>14}ms "
          f"{result_with_cache['cached_run_avg_ms']:>14}ms")
    print(f"{'重复查询加速比':<25} {result_no_cache['speedup']:>14}x "
          f"{result_with_cache['speedup']:>14}x")
    
    print(f"\n检索缓存命中率: {result_with_cache['cache_stats']['retrieval_hit_rate']}")


if __name__ == "__main__":
    compare_cached_vs_uncached()
```

---

## 三、知识总结

### 缓存策略选型

| 场景 | 推荐方案 | 预期节省 |
|------|---------|---------|
| 开发/测试 | `InMemoryCache` | 减少等待 |
| 生产 + 精确匹配 | `RedisCache` | 30-50% |
| 生产 + 语义相似 | `RedisSemanticCache` | 50-70% |
| Embedding 复用 | `CacheBackedEmbeddings` | 80%+ |
| 完整 RAG 优化 | 多级组合 | 60-80% |

### 面试高频问题

1. **语义缓存和精确缓存的区别与适用场景？**
   - 精确缓存：完全相同才命中，精度100%，适合FAQ；语义缓存：相似语义也命中，适合开放式问答

2. **缓存对 Agent 的影响有哪些负面效果？**
   - 工具结果缓存可能返回过期数据（如实时价格）；需要按工具类型区分是否缓存

3. **生产环境中如何实现缓存的高可用？**
   - Redis Sentinel / Redis Cluster，设置合理 TTL，实现 Cache-Aside 或 Write-Through 模式
