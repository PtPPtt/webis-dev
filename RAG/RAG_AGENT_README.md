# Webis RAG Agent 框架说明

该小型框架实现了一个条件触发（conditional）RAG Agent，用于在接收用户 Query 时：

- 优先使用本地 RAG 存储的先验知识；
- 当本地 RAG 未覆盖（相似度低于阈值）或显式要求刷新时，调用外部 `webis_fetcher` 获取新的清理文本并加入 RAG；
- 将检索到的上下文以特定 prompt 形式注入到 Agent（抽取/回答器），返回最终结果。

主要组件

- `structuring/rag_agent.py`：实现 `WebisRAGAgent`，包含 `make_pipeline_fetcher()` 工厂。
- `structuring/rag_component.py`、`structuring/rag_tools.py`：轻量级 RAG 存储与管理。
- `structuring/rag_pipeline_integration.py`：提供 CLI（含 `agent` 子命令）用于交互式调用。
- `structuring/rag_agent_demo.py`：无需 LLM 的本地演示（使用 stub fetcher）。

快速开始（无需 LLM）

```bash
python -m structuring.rag_agent_demo
```

使用真实管线输出注入 fetcher

假设你的管线输出目录结构是：

```
output_dir/
  extracted_texts/
    source1_clean.txt
    source2_clean.txt
  structured_data/
    source1_structured.json
    source2_structured.json
```

可通过 CLI 运行 agent（需已配置 LLM）：

```bash
python -m structuring.rag_pipeline_integration agent --query "你的问题" --pipeline-output /path/to/output_dir
```

或在代码中创建 agent：

```python
from structuring.rag_agent import WebisRAGAgent, make_pipeline_fetcher
from structuring.llm import get_default_llm

llm = get_default_llm()
fetcher = make_pipeline_fetcher('/path/to/output_dir')
agent = WebisRAGAgent(llm=llm, webis_fetcher=fetcher)
res = agent.handle_query('查询文本')
```

可调参数

- `min_score_threshold`（默认 0.35）：判断是否需要触发 webis 的相似度阈值。
- `top_k`：RAG 检索的文档数。

下一步建议

- 若需更高检索精度，可替换或扩展 `SimpleVectorStore` 为 Faiss/Annoy/Chroma 等；
- 若希望持久化并共享索引，请指定 `rag_store_path` 为可访问路径；
- 我可以帮你把真实的 webis 请求逻辑（API 或本地爬虫）实现为 `webis_fetcher`。
