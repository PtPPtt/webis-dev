# Webis: AI 驱动的数据处理流水线

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Webis** 是一个模块化、插件优先的 AI 数据框架。它致力于将互联网海量数据（Web, SaaS, 数据库）连接到大语言模型（LLM），通过智能的获取、清洗、抽取流程，为 AI 应用提供高质量的上下文知识。

## 🚀 核心特性

* **插件化架构 (Plugin-First)**: 一切皆插件（数据源、处理器、提取器、模型），高度可扩展。
* **智能爬虫 Agent**: 内置基于 LLM 的 `CrawlerAgent`，能理解自然语言任务，自动选择最佳数据源并生成查询。
* **RAG 原生支持**: 内置 PDF/HTML 清洗、降噪、切片功能，专为 RAG 知识库构建设计。
* **LLM 结构化抽取**: 通过动态 Schema 推断，将非结构化文档转化为标准 JSON 数据。
* **统一 CLI**: 一个 `webis` 命令搞定所有流程。

## 📦 安装指南

```bash
cd webis
pip install -e .
```

## 🛠️ 使用指南

Webis 通过 `webis` 命令行工具进行操作。

### 1. 端到端运行 (Run)

执行完整流水线：自动规划数据源 -> 抓取 -> 清洗 -> 结构化抽取。

```bash
# 例子：查找北京大学近三个月的新闻并生成报告
webis run "查找北京大学近三个月的新闻并生成报告" --limit 3
```

✨ **New**: 结果会自动保存到 `output/{timestamp}/` 目录，包含 JSON 数据和精美的 HTML 报告网页。

### 2. 仅抓取 (Crawl)

只进行搜索和下载，返回包含文档内容的 JSON 列表。

```bash
webis crawl "Python 3.13 新特性" --limit 5 -o crawl_results.json
```

### 3. 本地抽取 (Extract)

对本地文件（PDF, Word, TXT, MD）进行 AI 结构化提取。

```bash
# 从研报 PDF 中提取摘要
webis extract ./report.pdf --task "提取财务摘要和风险提示"

# 使用指定 Schema 进行提取
webis extract ./resume.pdf --schema ./schemas/resume_schema.json
```

## 🧩 系统架构

项目核心代码位于 `src/webis/`：

* **`core/`**: 内核层 (Agent 基类, Pipeline 引擎, Plugin 注册表)。
* **`plugins/`**:
  * `sources/`: 数据源插件 (GNews, Google Search, GitHub 等)。
  * `processors/`: 数据处理器 (PDF 解析, HTML 清洗等)。
  * `extractors/`: 抽取器 (LLMExtractor)。
* **`plugin_sdk/`**: 开发者工具包，用于快速开发新插件。

## 🤝 参与贡献

欢迎提交 PR！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何使用 SDK 开发新插件。
