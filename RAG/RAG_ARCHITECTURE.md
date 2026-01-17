# Webis RAG 系统架构说明文档

## 📋 概览

Webis RAG（检索增强生成）系统是一个完整的文档获取、处理、向量化和检索框架。系统包含四个核心模块，协作实现从原始数据到智能问答的完整流程。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Webis RAG System Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Query                                                      │
│      │                                                            │
│      ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         WebisRAGAgent (rag_agent.py)                    │    │
│  │  • 决策：是否需要调用管道获取数据                        │    │
│  │  • 管理：问答流程，存储结果                              │    │
│  └──────────────────┬──────────────────────────────────────┘    │
│                     │                                            │
│         ┌───────────┴──────────────┐                            │
│         │ Need more data?          │                            │
│         ▼                          ▼                            │
│   ┌──────────────┐         ┌─────────────────┐                │
│   │ Call Pipeline│         │ Query RAG Store │                │
│   └──────────────┘         └─────────────────┘                │
│         │                          ▲                            │
│         ▼                          │                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Webis Pipeline (Internal)                              │    │
│  │  ┌────────┐  ┌──────┐  ┌─────┐  ┌──────┐  ┌───────┐   │    │
│  │  │ Search │→ │Fetch │→ │Clean│→ │Chunk │→│Embed  │   │    │
│  │  └────────┘  └──────┘  └─────┘  └──────┘  └───────┘   │    │
│  │                                        │                  │    │
│  │                                        ▼                  │    │
│  │                                  ┌──────────┐           │    │
│  │                                  │ Extract  │           │    │
│  │                                  │Structured│           │    │
│  │                                  └──────────┘           │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                       │
│         ┌───────────────┴────────────────┐                     │
│         │                                │                     │
│         ▼                                ▼                     │
│  ┌─────────────────────┐      ┌──────────────────────────┐   │
│  │ RAGManager          │      │ Save WebisDocuments      │   │
│  │ (rag_tools.py)      │      │ to JSON                  │   │
│  │ • Add documents     │      │ data/webis_documents/    │   │
│  │ • Build index       │      └──────────────────────────┘   │
│  │ • Retrieve results  │                                     │
│  └──────────┬──────────┘                                     │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ RAGComponent + SimpleVectorStore (rag_component.py)  │   │
│  │ • 向量存储                                            │   │
│  │ • 向量检索（BM25 + 余弦相似度）                      │   │
│  │ • Embedding 优先级：外部→TF-IDF→Hash               │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Enhanced Prompt + Retrieved Documents               │   │
│  │ → Send to Agent/LLM for final answer                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 核心模块详解

### 1️⃣ WebisRAGAgent (rag_agent.py) - 智能决策和流程管理

**职责**：
- 根据查询决定是否需要调用管道获取新数据
- 管理完整的问答流程
- 保存 WebisDocument 对象为 JSON 格式

**核心方法**：

#### `__init__(llm, rag_store_path, top_k, min_score_threshold)`
```
初始化 RAG Agent
- llm: LLM 实例（用于答案生成）
- rag_store_path: 向量存储路径
- top_k: 检索结果数量
- min_score_threshold: 相似度阈值（0.30）
  • 若最高相似度 ≥ 阈值：认为 RAG 已覆盖该领域，无需调用管道
  • 若最高相似度 < 阈值：需要调用管道获取新数据
```

#### `should_fetch_webis(query) -> bool`
```
决策函数：是否需要调用管道

流程：
  1. 从 RAG 检索 top_k 个文档
  2. 获取检索得分
  3. 若无结果或最高得分 < min_score_threshold，返回 True
  4. 否则返回 False
```

#### `fetch_and_add(query) -> List[str]`
```
调用管道并添加到 RAG

流程：
  1. 执行完整 Webis Pipeline
  2. 提取清洁文本、结构化数据、Embedding
  3. 添加到 RAG Manager
  4. 返回新增文档 ID 列表
```

#### `_save_webis_documents(documents, query) -> None`
```
保存 WebisDocument 对象为 JSON

输出格式：
  data/webis_documents/webis_docs_[query_safe]_[YYYYMMDD_HHMMSS].json

JSON 结构：
  {
    "query": "用户查询",
    "timestamp": "2026-01-18T10:30:45",
    "document_count": 3,
    "documents": [
      {
        "id": "doc-uuid",
        "content": "完整原始内容",
        "clean_content": "清洗后的内容",
        "doc_type": "html",
        "status": "completed",
        "metadata": {
          "url": "source_url",
          "title": "文档标题",
          "author": "作者",
          "source_plugin": "DuckDuckGo",
          "language": "en",
          "tags": [],
          "custom": {}
        },
        "chunks": [
          {
            "id": "chunk-uuid",
            "content": "块内容",
            "index": 0,
            "embedding": [0.1, 0.2, ...],
            "metadata": {}
          }
        ],
        "chunks_count": 5,
        "processing_history": [
          {
            "plugin": "HtmlFetcher",
            "timestamp": "2026-01-18T10:30:40",
            "details": {}
          }
        ],
        "parent_id": null
      }
    ]
  }
```

#### `handle_query(query, extraction_goal, output_format, force_refresh, use_webis_if_needed) -> Dict`
```
主入口：完整的问答流程

参数说明：
  - query: 用户查询
  - extraction_goal: 提取目标（默认"基于参考信息回答用户查询"）
  - output_format: 输出格式（markdown/json）
  - force_refresh: 强制刷新（跳过相似度检查）
  - use_webis_if_needed: 是否按需调用管道

流程：
  1. 判断是否需要调用管道（基于 should_fetch_webis）
  2. 若需要则调用管道获取数据
  3. 使用 RAG 检索相关文档
  4. 增强提示词（添加检索到的文档上下文）
  5. 调用 Agent/LLM 生成答案
  6. 保存答案到 JSON 文件

返回结果：
  {
    "query": "用户查询",
    "used_webis": true,
    "retrieved_documents": ["source1", "source2"],
    "enhanced_prompt": "增强的提示词...",
    "agent_result": {
      "success": true,
      "output_format": "markdown",
      "parsed": {...},
      "raw": "原始输出",
      "error": null
    }
  }
```

---

### 2️⃣ Webis Pipeline (rag_agent.py 内部) - 数据处理管道

**职责**：完整的数据处理流程，从搜索到结构化提取

**处理阶段**：

```
┌──────────────┐
│ DuckDuckGo   │ (搜索源)
│ Search       │
└──────┬───────┘
       │ URL 列表
       ▼
┌──────────────┐
│ HtmlFetcher  │ (获取 HTML)
└──────┬───────┘
       │ HTML 内容
       ▼
┌──────────────┐
│ HtmlCleaner  │ (清洗 HTML)
└──────┬───────┘
       │ 清洁文本
       ▼
┌──────────────┐
│ Chunking     │ (分块处理)
└──────┬───────┘
       │ 文本块 + 元数据
       ▼
┌──────────────┐
│ Embedding    │ (向量化)
│ (OpenAI)     │ • 使用 text-embedding-3-small
└──────┬───────┤ • 1536 维向量
       │ • 添加到每个 chunk
       ▼
┌──────────────┐
│ LLMExtractor │ (结构化提取)
│              │ • 提取标题、关键词、实体、话题
└──────────────┘
       │
       ▼
  WebisDocument 对象列表
```

**各阶段输出**：

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| Search | 查询 | URL列表 | DuckDuckGo 搜索结果 |
| Fetch | URL | HTML | 获取原始 HTML 内容 |
| Clean | HTML | 清洁文本 | 移除脚本、样式、标签 |
| Chunk | 文本 | DocumentChunk[] | 分块，保留索引和边界信息 |
| Embed | Chunk[] | embedding[] | OpenAI 向量化，1536维 |
| Extract | clean_text | StructuredResult | JSON 格式的结构化数据 |

---

### 3️⃣ RAGManager & RAGComponent (rag_tools.py + rag_component.py) - 向量存储与检索

**职责**：
- 管理文档存储和检索
- 智能 Embedding 策略
- 提示词增强

#### RAGManager (rag_tools.py) - 高层管理器

```python
RAGManager.__init__(
    rag_store_path="./RAG/rag_store.json",
    auto_load=True,
    use_external_embeddings=True  # ← 关键参数
)
```

**Embedding 优先级策略**：
```
1. 外部 Embedding（推荐）
   └─ 来自 EmbeddingPlugin (OpenAI text-embedding-3-small, 1536维)
   
2. TF-IDF（按需初始化的后备方案）
   └─ 仅当无外部 embedding 时动态初始化
   └─ 保存节省内存，不提前加载 sklearn
   
3. Hash-based（最后的备选）
   └─ 当 sklearn 不可用时使用
```

**核心方法**：

```python
def add_crawled_documents(documents: List[Dict]) -> List[str]
    """批量添加爬虫获得的文档
    
    参数示例：
    [
        {
            "content": "清洁文本",
            "source": "https://example.com",
            "structured_data": {"title": "...", "entities": [...]},
            "embeddings": [[0.1, 0.2, ...], ...],  # 多个 chunk 的 embedding
            "chunks": ["chunk1", "chunk2"],
            "metadata": {"from_pipeline": True}
        }
    ]
    
    处理流程：
    1. 若有多个 embeddings，计算平均值
    2. 添加到向量存储
    3. 返回文档 ID 列表
    """

def retrieve_for_query(query: str, top_k=3, include_scores=False) -> Dict
    """检索查询的相关文档
    
    返回结果：
    {
        "query": "用户查询",
        "documents": [
            {
                "source": "source_url",
                "content": "文档内容（截断200字）",
                "structured_data": {...}
            }
        ],
        "context": "格式化的上下文文本",
        "scores": [0.85, 0.72, 0.65]  # 若 include_scores=True
    }
    """

def enhance_agent_prompt(base_prompt: str, user_query: str, rag_top_k=3) -> Tuple
    """增强提示词：添加检索到的文档上下文
    
    返回：(增强的提示词, 上下文数据)
    """

def build_and_save()
    """构建索引并保存到磁盘
    
    1. build_index() - 计算/验证向量
    2. save() - 持久化存储
    """
```

#### SimpleVectorStore (rag_component.py) - 轻量级向量存储

```python
class SimpleVectorStore:
    """
    内存向量存储 + 本地序列化
    
    特点：
    - 适合小规模（<10k 文档）
    - 支持外部 embedding 或 TF-IDF
    - 自动维度对齐（重要！）
    """

    def __init__(self, use_external_embeddings=True)
        """初始化存储
        
        - True: 使用外部 embedding（推荐），不提前初始化 TF-IDF
        - False: 使用 TF-IDF 作为主方案
        """

    def _init_tfidf_vectorizer()
        """延迟初始化 TF-IDF
        
        仅在无外部 embedding 时按需加载 sklearn
        减少内存占用和初始化时间
        """

    def build_index()
        """构建索引
        
        逻辑：
        1. 检查文档是否已有 embedding
        2. 若全部有 embedding，跳过计算
        3. 否则按需初始化 TF-IDF，计算向量
        4. 最后备选：Hash-based 向量
        """

    def retrieve(query: str, top_k=5) -> List[Tuple[RAGDocument, float]]
        """检索相关文档
        
        步骤：
        1. 对查询生成 embedding
           - 尝试使用 OpenAI embedding
           - 失败则回退到 TF-IDF
        2. 计算查询与文档的余弦相似度
        3. 自动维度对齐（处理不同维度的向量）
        4. 按相似度排序，返回 top_k
        
        维度对齐示例：
        - 外部 embedding：1536 维
        - TF-IDF：100 维
        → 自动截取最小维度后比较
        """

    def save(path: str)
        """序列化到磁盘
        
        保存内容：
        1. documents.json - 文档和向量
        2. vectorizer.pkl - TF-IDF 向量化器（若使用）
        """

    def load(path: str)
        """从磁盘加载
        
        恢复所有文档和向量化器
        """
```

---

### 4️⃣ WebisDocument (schema.py) - 数据模型

**职责**：定义文档数据模型，支持 JSON 序列化

```python
class WebisDocument(BaseModel):
    """
    Webis 系统的核心数据单元
    """
    
    # 基本字段
    id: str                          # 唯一 UUID
    content: str                     # 原始内容
    clean_content: Optional[str]     # 清洁后的内容
    doc_type: DocumentType           # html/pdf/text/markdown/...
    status: DocumentStatus           # pending/processing/completed/failed
    
    # 元数据
    meta: DocumentMetadata
        └─ url, title, author
        └─ published_at, fetched_at
        └─ source_plugin, language
        └─ tags, custom
    
    # 处理结果
    chunks: List[DocumentChunk]      # 文本块
        └─ content, index, metadata
        └─ embedding (可选)
    embeddings: List[List[float]]    # 所有 chunk embeddings
    
    # 处理历史
    processing_history: List[Dict]
        └─ plugin, timestamp, details
    
    parent_id: Optional[str]         # 衍生文档的父 ID
    
    def to_json(include_embeddings=True) -> Dict
        """转换为 JSON 序列化格式
        
        功能：
        - 保存所有完整信息（不截断）
        - 包含所有 chunks 详情
        - 默认包含 embeddings（可选关闭以减小文件）
        - 时间戳转为 ISO 格式
        """
```

**DocumentType 枚举**：
```
HTML, PDF, TEXT, MARKDOWN, IMAGE, AUDIO, VIDEO, JSON, UNKNOWN
```

**DocumentStatus 枚举**：
```
PENDING → PROCESSING → COMPLETED / FAILED / SKIPPED
```

---

## 📊 数据流示例

### 完整的查询-应答流程

```
输入：handle_query("最近 AI 芯片的新闻")

阶段 1: 决策
├─ 调用 should_fetch_webis(query)
├─ 从 RAG 检索已有文档
├─ 最高相似度 = 0.25 < 阈值(0.30)
└─ 返回 True → 需要调用管道

阶段 2: 管道执行
├─ DuckDuckGo 搜索 → 3 个 URL
├─ HtmlFetcher 获取 → 3 个 HTML
├─ HtmlCleaner 清洁 → 3 个清洁文本
├─ Chunking 分块 → 15 个 chunk
├─ Embedding 向量化 → 15 × 1536 维向量
└─ LLMExtractor 提取 → 3 个 StructuredResult

阶段 3: 保存
├─ _save_webis_documents() 
└─ → data/webis_documents/webis_docs_AI_20260118_103045.json
   │  包含：3 个完整 WebisDocument（所有字段）

阶段 4: RAG 添加
├─ RAGManager.add_crawled_documents()
├─ 添加 3 个文档到向量存储
├─ build_index() 使用外部 embedding
└─ 保存到 data/rag_store.json

阶段 5: 检索增强
├─ retrieve_for_query("最近 AI 芯片的新闻")
├─ 查询 embedding 由 OpenAI 生成
├─ 计算余弦相似度 → [0.92, 0.88, 0.85]
├─ 返回 top-3 文档
└─ 格式化为上下文

阶段 6: 提示词增强
├─ 原始提示词 + 检索到的文档
└─ Enhanced Prompt:
   """
   ## 参考信息（来自知识库）
   
   ### 检索的文档：
   [Source: https://...] (Relevance: 0.92)
   AI 芯片...
   
   ### 结构化数据：
   - 标题: ...
   - 关键词: ...
   
   ---
   
   ## 用户查询
   最近 AI 芯片的新闻
   """

阶段 7: Agent 调用
├─ 使用 Enhanced Prompt 调用 LLM
└─ 返回答案

输出：agent_result 保存到 data/agent_answer_20260118_103045.json
```

---

## 🔧 关键设计决策

### 1. Embedding 优先级策略

**为什么？**
- 外部 embedding（OpenAI）质量最好，但需要 API 调用
- TF-IDF 是本地、快速的后备方案
- Hash-based 是最后的保障

**实现**：
```
初始化 SimpleVectorStore(use_external_embeddings=True)
├─ 不提前初始化 TF-IDF（节省内存）
└─ build_index() 时检查文档是否已有 embedding
   ├─ 若全部有 → 直接使用（最快）
   └─ 若无 → 按需初始化 TF-IDF
```

### 2. 自动维度对齐

**为什么？**
- 外部 embedding：1536 维
- TF-IDF：100 维（可配置）
- 查询可能用不同方案生成，维度可能不一致

**实现**：
```python
# retrieve() 中
min_dim = min(len(doc_emb), len(query_emb))
doc_emb = doc_emb[:min_dim]
query_emb = query_emb[:min_dim]
# 然后计算余弦相似度
```

### 3. 延迟初始化

**为什么？**
- sklearn 是可选依赖，可能不装
- 没有外部 embedding 时才真正需要 TF-IDF
- 减少启动时间和内存占用

**实现**：
```python
def _init_embedding_scheme(self):
    # 只记录使用外部 embedding
    if self.use_external_embeddings:
        logger.info("Using external embeddings...")
    # 不初始化 TF-IDF

def _init_tfidf_vectorizer(self):
    # 真正需要时才初始化
    from sklearn.feature_extraction.text import TfidfVectorizer
    self.vectorizer = TfidfVectorizer(...)
```

### 4. 完整数据保存

**为什么？**
- 审计：追踪数据处理的完整历史
- 重用：无需重新爬取，可直接用已保存数据
- 调试：检查各阶段输出

**实现**：
```python
WebisDocument.to_json(include_embeddings=True)
# 保存所有字段：
# - 完整 content/clean_content（不截断）
# - 所有 chunks 含 embedding
# - 完整处理历史
# - 所有元数据
```

---

## 📁 文件结构

```
webis-dev/
├── RAG/
│   ├── rag_agent.py              # 核心 Agent，管道调用，文档保存
│   ├── rag_component.py          # 向量存储，检索，提示词增强
│   ├── rag_tools.py              # RAGManager，高层接口
│   ├── rag_agent_demo.py         # 使用示例
│   └── rag_store.json            # 向量存储持久化文件
│
├── src/webis/core/
│   └── schema.py                 # 数据模型定义
│
├── data/
│   ├── rag_store.json            # RAG 向量存储
│   ├── webis_documents/          # WebisDocument JSON 备份
│   │   ├── webis_docs_query1_20260118_103045.json
│   │   └── webis_docs_query2_20260118_103050.json
│   └── agent_answer_*.json       # Agent 答案备份
│
└── src/webis/
    ├── plugins/
    │   ├── sources/
    │   │   └── DuckDuckGoPlugin  # 搜索源
    │   ├── processors/
    │   │   ├── HtmlFetcherPlugin
    │   │   ├── HtmlCleanerPlugin
    │   │   ├── ChunkingPlugin
    │   │   └── EmbeddingPlugin   # OpenAI 向量化
    │   └── extractors/
    │       └── LLMExtractorPlugin # 结构化提取
    └── core/
        └── pipeline.py            # Pipeline 框架
```

---

## 🚀 使用示例

### 基础使用

```python
from RAG.rag_agent import WebisRAGAgent

# 创建 Agent
agent = WebisRAGAgent(
    llm=None,  # 本地调试不需要 LLM
    rag_store_path="./data/rag_store.json"
)

# 执行查询
result = agent.handle_query(
    query="最近 AI 芯片的新闻",
    force_refresh=False,  # 按需调用管道
    use_webis_if_needed=True
)

# 查看结果
print(f"使用 Webis: {result['used_webis']}")
print(f"检索文档: {result['retrieved_documents']}")
print(f"增强提示词:\n{result['enhanced_prompt']}")
if result['agent_result']['success']:
    print(f"答案: {result['agent_result']['parsed']}")
```

### 强制刷新

```python
# 跳过相似度检查，强制调用管道
result = agent.handle_query(
    query="AI 芯片",
    force_refresh=True  # 即使 RAG 有高相似度文档也重新获取
)
```

### 禁用 Webis

```python
# 仅使用已有 RAG 数据，不调用管道
result = agent.handle_query(
    query="AI 芯片",
    use_webis_if_needed=False  # 不调用管道
)
```

---

## 📈 性能特点

| 指标 | 说明 |
|------|------|
| **存储容量** | <10k 文档，内存存储 |
| **检索速度** | 毫秒级（内存查询） |
| **向量维度** | 1536（OpenAI）或 100（TF-IDF）或 128（Hash） |
| **相似度** | 余弦相似度 |
| **BM25** | 不使用（使用向量为主） |
| **更新** | 增量添加，需要 rebuild_index |

---

## 🔍 故障排查

### 问题 1：检索效果差

**可能原因**：
1. Embedding 质量差（使用了 Hash-based）
2. 相似度阈值设置过高
3. 查询与文档不匹配

**解决**：
```python
# 检查 embedding 类型
stats = agent.rag_manager.get_stats()
print(stats)  # 查看索引状态

# 调整阈值
agent.min_score_threshold = 0.20  # 降低阈值

# 强制重新获取数据
result = agent.handle_query(query, force_refresh=True)
```

### 问题 2：内存使用高

**可能原因**：
1. 保存了大量 embedding
2. 文档内容过长

**解决**：
```python
# 保存时不包含 embedding
doc.to_json(include_embeddings=False)

# 定期清理旧数据
import shutil
shutil.rmtree("data/webis_documents/")
```

### 问题 3：管道执行失败

**检查步骤**：
```python
# 1. 检查管道日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. 检查 Fallback 文档保存
# data/webis_documents/webis_docs_fallback_*.json

# 3. 验证 pipeline 插件
from webis.core.pipeline import Pipeline
pipe = Pipeline()
print(pipe.registry.list_plugins())
```

---

## 📝 总结

Webis RAG 系统通过 **分层设计** 实现：

1. **WebisRAGAgent** - 智能决策层
   - 判断何时需要新数据
   - 管理完整工作流
   
2. **Webis Pipeline** - 数据处理层
   - 搜索 → 获取 → 清洁 → 分块 → 向量化 → 提取
   
3. **RAGManager + SimpleVectorStore** - 存储检索层
   - 智能 embedding 优先级
   - 高效的向量相似度检索
   
4. **WebisDocument** - 数据模型层
   - 统一的数据表示
   - 完整的信息保存

系统的核心优势是 **智能决策** 和 **完整追踪**，确保在获取新数据和使用已有数据之间取得平衡，同时保留完整的处理历史供审计和重用。
