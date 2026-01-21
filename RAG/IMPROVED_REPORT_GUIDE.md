# 精炼报告生成 - 改进指南

## 问题与改进

### 原问题
- 报告内容冗余
- 原文信息过多
- 缺少来源标注
- LLM Prompt 不够结构化

### 改进方案
✓ 精炼的信息提取（只提取关键内容）
✓ 明确的来源标注（每条信息标注【来源：文档名】）
✓ 结构化的 LLM Prompt
✓ 减少冗余信息

## 核心改进

### 1. 信息来源标注

所有报告中的信息都遵循以下格式：

```markdown
【来源：文档名】关键信息内容
```

#### 在各部分中的体现

**执行摘要:**
```
【来源：AI芯片2024进展】最新AI芯片采用3nm工艺
【来源：NVIDIA官方】H100性能提升50%
```

**详细发现:**
```
【来源：云计算报告】云原生应用占比达70%
【来源：Gartner分析】Kubernetes采用率持续增长
```

**关键发现:**
```
- 【来源：量子计算论文】量子比特数达到1000+
- 【来源：IBM公告】2年内实现商用量子计算
```

### 2. 精炼的内容提取

#### 执行摘要（3个关键点）
```python
def _generate_summary(self, query: str, documents: List[Dict]) -> str:
    # 只提取Top 5文档
    doc_chunks = []
    for doc in documents[:5]:
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")[:300]  # 每篇限300字
        doc_chunks.append(f"【来源：{source}】\n{content}")
```

#### 详细发现（Top 3文档）
```python
def _generate_detailed_content(self, query: str, documents) -> str:
    # 只分析Top 3文档
    for doc in documents[:3]:
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")[:400]
```

#### 关键发现（5个要点）
```python
def _extract_key_findings(self, query: str, documents) -> List[str]:
    # 生成不超过5个关键发现
    # 每条一句话，带来源标注
```

### 3. LLM Prompt 结构化

#### 摘要生成 Prompt

```python
prompt = f"""根据以下检索到的文档信息，为查询提供简洁的要点总结（2-3个关键点，每点1-2句话）。
每个要点必须标注来源文档。

查询：{query}

文档信息：
{docs_text}

要点总结格式：
- 【来源：文档名】关键要点内容
- 【来源：文档名】关键要点内容

总结："""
```

#### 详细分析 Prompt

```python
prompt = f"""基于以下检索到的文档，针对查询进行结构化分析。请：
1. 提取关键信息点（精炼，去除冗余）
2. 指出不同文档的观点差异（如有）
3. 总结主要趋势或结论
每条信息必须标注来源。

查询：{query}

文档内容：
{docs_text}

分析格式：
【来源：文档名】信息内容
【来源：文档名】信息内容

详细分析："""
```

#### 关键发现提取 Prompt

```python
prompt = f"""从以下文档中提取关于"{query}"的5个最关键的发现或要点。
要求：
1. 每个发现必须精炼简洁（1句话）
2. 每个发现必须标注来源文档
3. 避免重复或通用表述
4. 按重要性排序

文档：
{docs_text}

关键发现列表格式：
【来源：文档名】发现内容

关键发现："""
```

## 报告结构

```
# 研究报告

## 概览
- 检索文档数
- Webis获取
- 最高相关度

## 执行摘要
【来源：文档1】摘要1
【来源：文档2】摘要2
【来源：文档3】摘要3

## 详细发现
### 【来源：文档1】
关键内容段落

### 【来源：文档2】
关键内容段落

## 关键发现
- 【来源：文档1】发现1
- 【来源：文档2】发现2
- 【来源：文档3】发现3

## 信息来源
[1] 文档名 (相关度: 0.95)
[2] 文档名 (相关度: 0.89)
...

## 元数据
```

## 使用示例

### 基础用法

```python
from rag_pipeline import RAGPipeline
from rag_tasks import TaskPipeline, ReportGenerationTask
from langchain_openai import ChatOpenAI

# 初始化LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")

# 初始化RAG
rag = RAGPipeline(min_doc_threshold=1, min_score_threshold=0.2)

# 创建任务
task = ReportGenerationTask(
    llm=llm,
    include_raw_data=False,  # 只要精炼内容
    output_format="markdown"
)

# 生成报告
query = "AI芯片最新进展"
context = rag.get_retrieval_context(query, auto_fetch_webis=True)
task_pipeline = TaskPipeline()
task_pipeline.add_task(task)
result = task_pipeline.execute(context)

# 报告已保存到 ./data/report_TIMESTAMP.md
```

### 批量生成

```python
queries = [
    "5G技术",
    "量子计算",
    "边缘计算"
]

for query in queries:
    context = rag.get_retrieval_context(query, auto_fetch_webis=True)
    result = task_pipeline.execute(context)
    print(f"✓ 生成报告: {result['task_results'][0]['output_path']}")
```

## 参数说明

### ReportGenerationTask 参数

```python
ReportGenerationTask(
    llm=None,                    # LLM实例（可选）
    include_raw_data=False,      # 是否包含原始数据
    output_format="markdown"     # 输出格式：markdown 或 pdf
)
```

### 效果对比

| 方面 | 无LLM | 有LLM |
|------|-------|-------|
| 摘要来源 | 直接提取 | LLM综合 |
| 发现结构 | 简单列表 | 结构化分析 |
| 冗余度 | 可能有冗余 | 精炼 |
| 观点对比 | 无 | 有 |
| 质量 | 基础 | 优质 |

## 演示脚本

运行改进报告生成演示：

```bash
cd RAG
python demo_improved_report.py
```

### 包含的演示

1. **Demo 1**: 精炼报告 - 带来源标注
2. **Demo 2**: 来源追踪 - 查看每条信息的来源
3. **Demo 3**: 有无LLM对比
4. **Demo 4**: 批量生成 - 多个查询

## 最佳实践

### 1. 选择合适的 LLM
```python
# 推荐：成本效益好
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# 高质量（更贵）
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
```

### 2. 调整文档数量
```python
# 根据查询复杂度调整：
# 简单查询：Top 3 文档
# 复杂查询：Top 5-7 文档
# 综合查询：Top 10 文档
```

### 3. 验证来源标注
```python
# 检查报告中是否每条信息都有【来源：...】
# 验证来源与原始文档对应
```

### 4. 性能优化
```python
# 减少不必要的字段
include_raw_data=False  # 默认False，保持精炼

# 使用更快的模型
model="gpt-3.5-turbo"  # 如果质量要求不高
```

## 故障排查

### 问题1: 来源标注不完整

**原因**: LLM 没有按格式输出

**解决**:
```python
# 增强 prompt 中的指令
prompt += "\n必须每条信息都以【来源：文档名】开头"
```

### 问题2: 信息仍然冗余

**原因**: 限制不够严格

**解决**:
```python
# 减少文档数量
for doc in documents[:3]:  # 改为3而不是5

# 减少字符数
content = doc.get("content", "")[:200]  # 改为200
```

### 问题3: LLM 调用失败

**原因**: API 错误或限制

**解决**:
```python
# 检查 API Key
print(os.getenv("OPENAI_API_KEY"))

# 添加重试逻辑
from tenacity import retry, stop_after_attempt
@retry(stop=stop_after_attempt(3))
def call_llm(...):
    ...
```

## 输出示例

### 执行摘要

```
【来源：AI芯片2024年突破】NVIDIA发布H200芯片，搭载92GB HBM3E显存
【来源：芯片工艺进展】英特尔完成4A工艺认证，成本下降30%
【来源：ARM最新声明】ARM V9架构芯片出货量超过50亿片
```

### 关键发现

```
- 【来源：Gartner报告】2024年AI芯片市场规模突破200亿美元
- 【来源：OpenAI技术文章】Transformer架构优化使推理速度提升3倍
- 【来源：Meta研究论文】混合精度训练减少80%显存占用
- 【来源：Google官方】TPU性能提升迭代继续推进
- 【来源：行业分析】国产芯片自给率达到15%
```

## 总结

改进后的报告生成系统具有以下优势：

✓ **精炼**: 无冗余，精确提取关键信息
✓ **可追踪**: 每条信息都有【来源：...】标注
✓ **结构化**: LLM Prompt 指导精准输出
✓ **灵活**: 支持有无 LLM 两种模式
✓ **可扩展**: 易于调整参数和 prompt

