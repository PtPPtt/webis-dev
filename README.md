# Webis: From Web Data Crawling to Thematic Knowledge Base in an Efficient Pipeline

[English](README.md) | [Chinese](README_CN.md)

![Python Version](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-Open%20Source-green)

## Project Overview

**Webis** is a powerful end-to-end data processing platform that combines **large language models (LLMs)** with **crawler tooling**. It provides a high-efficiency, automated pipeline for data processing. The platform helps users quickly collect unstructured web data, clean it, and produce structured outputs such as JSON or Markdown reports, enabling rapid construction of thematic knowledge bases.

In today's information explosion, data and content are generated constantly. Traditional data processing is often slow and inefficient. Webis uses an intelligent pipeline design to greatly improve the efficiency of data collection, cleaning, and structuring. It also supports **customization** and **flexible extensibility**, so it can adapt to different application scenarios.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
   - [System Requirements](#system-requirements)
   - [Installation](#installation)
   - [Configure API Keys](#configure-api-keys)
4. [Main Directory Structure](#main-directory-structure)
5. [Usage](#usage)
   - [CLI Parameters](#cli-parameters)
   - [Outputs and Directory Structure](#outputs-and-directory-structure)
   - [Example 1: Layoff Information from Chinese Enterprises](#example-1-layoff-information-from-chinese-enterprises)
   - [Example 2: Recent News from Peking University](#example-2-recent-news-from-peking-university)
6. [Development and Extensions](#development-and-extensions)
7. [Contributing](#contributing)

---

## Key Features

- **Intelligent Data Processing and Structured Extraction**
  Webis provides a three-stage pipeline: data crawling -> text cleaning -> structured output. It can deeply extract content from news, reports, and articles to produce structured knowledge bases such as JSON and Markdown reports.

- **Plugin Architecture with Flexible Extensions**
  The system includes multiple built-in crawler tools and supports custom crawler tools and parsers as plugins. You can extend it to fit different data sources and formats.

- **Multi-Domain Application Examples**
  Webis supports data collection and processing across domains. Two typical examples are:

  - **Enterprise Status Analysis**: Users can request data about layoffs from domestic companies and generate structured reports, including company names, layoff scale, time, and industry context.

  - **Academic News and Event Reports**: For example, the system can crawl recent news from Peking University, automatically extract academic events, innovation contests, awards, and summarize them into structured reports for quick overview.

---

## Quick Start

### System Requirements

- Python **3.9 or 3.10**
- Environment management with **Conda** or **UV**
- Git access
- LLM API Key (e.g., SiliconFlow / DeepSeek)

### Installation

1. **Install with Conda**

   ```bash
   git clone https://github.com/Easonnoway/webis-dev.git
   cd webis-dev
   bash setup/conda_setup.sh
   conda activate webis
   ```

2. **Install with UV**

   ```bash
   git clone https://github.com/Easonnoway/webis-dev.git
   cd webis-dev
   bash setup/uv_setup.sh
   source webis/bin/activate
   ```

### Configure API Keys

#### Primary LLM API Key

- `SILICONFLOW_API_KEY` (recommended) or `DEEPSEEK_API_KEY` (compatible)

#### Optional Crawler Tool API Keys

- The system will automatically choose tools based on available keys. Tools without keys will be disabled.

- News search
  - `GNEWS_API_KEY` - for GNewsTool
  - `SERPAPI_API_KEY` - for SerpApi Google Search
  - `BAIDU_AISEARCH_BEARER` - for Baidu AI Search

- Code search
  - `GITHUB_TOKEN` - for GitHub search

Make sure `.env.local` or `.env` contains the LLM API key(s):

```env
# required
SILICONFLOW_API_KEY=your_key_here
# optional
GNEWS_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here
BAIDU_AISEARCH_BEARER=your_key_here
GITHUB_TOKEN=your_token_here
```

---

## Main Directory Structure

```
webis-dev/
|-- crawler/                 # crawler tools and code
|-- tools/                   # file-type processors
|-- structuring/             # prompts and extraction templates
|-- setup/                   # installation and dependency scripts
|-- webis_pipeline.py        # main pipeline entry
|-- README.md
|-- .env.local
`-- pipeline_outputs/        # runtime outputs
```

---

## Usage

### CLI Parameters

Webis provides a set of command-line parameters to control execution. Main parameters:

| Option      | Type | Default                         | Description                                        |
| ----------- | ---- | ------------------------------- | -------------------------------------------------- |
| `task`      | str  | (required)                      | Natural language task for crawling and structuring |
| `--limit`   | int  | `5`                             | Maximum number of items to crawl                   |
| `--out`     | str  | `pipeline_outputs/<timestamp>/` | Custom output directory                            |
| `--verbose` | flag | `False`                         | Enable detailed logging                            |
| `--workers` | int  | `4`                             | Parallel workers for text extraction               |

### Outputs and Directory Structure

After execution, outputs are saved in a time-stamped directory:

```
pipeline_outputs/
`-- <timestamp>/                    # Run ID (Unix timestamp)
    |-- manifest.json               # Complete execution metadata
    |-- crawl_files.json            # List of acquired files
    |-- texts/                      # Cleaned text files
    |   |-- 0001_filename1.txt
    |   |-- 0002_filename2.txt
    |   `-- ...
    `-- structured/                 # Final structured output
        |-- prompt.txt              # Generated extraction prompt
        `-- result.{json|md}        # Structured result
```

### Structured Data Visualization

After the structured result is generated, you can use `visualization/visual.py` to convert the Webis pipeline output into a visual HTML page. The script saves the page to `pipeline_outputs/output_HTML/` and automatically opens it in the local browser.

Example:

```bash
python visualization/visual.py pipeline_outputs/<timestamp>/structured/result.json
```

Replace the path with your actual structured result file.

### Example 1: Layoff Information from Chinese Enterprises

Users can request crawling and analysis of **layoff information from Chinese enterprises**, and produce structured reports including company names, layoff scale, time, and more. Example command:

```bash
python webis_pipeline.py "Check recent layoffs in Chinese enterprises and output in structured form" --limit 10 --verbose
```

**Example Output (partial):**

```
[
  {
    "company_name": "JinkoSolar",
    "industry": "Photovoltaics",
    "layoff_scale": "5400 employees, layoff rate 6%",
    "layoff_time": "within the last half-year (2024-2025)",
    "notes": "Leading enterprise with significant workforce reduction"
  }
]
```

### Example 2: Recent News from Peking University

Users can request summarization of **recent news from Peking University** into a report format, covering academic events, innovation competitions, and other key items. Example command:

```bash
python webis_pipeline.py "Summarize recent news from Peking University" --limit 10 --verbose
```

**Example Output:**

```
Report on recent news from Peking University

Based on the provided text, the following is a summary of major recent news events:

1. Academic forums and discipline development
2. Innovation and entrepreneurship activities
3. Awards and scholarships
4. Scholar viewpoints and academic exchange
```

---

## Development and Extensions

### Extend Crawler Tools

Webis crawler components follow a unified plugin specification. To add a new data source, implement the **BaseTool interface**:

1. Define tool metadata (name, description, capabilities)
2. Implement the `run(task, **kwargs)` method to return standardized crawl results
3. Register the tool into the crawler agent for scheduling

This design ensures consistency across crawler implementations and enables flexible strategy switching or combination.

### Extend File Processors

All file-processing logic (HTML, PDF, image OCR, etc.) also uses a plugin pattern. To add a custom processor:

- Implement the `BaseFileProcessor` class and define supported file extensions
- Implement `extract_text()` to extract text content
- Return standardized fields such as `success`, `text`, and `meta`

This makes it easy to add support for new file types or specialized cleaning strategies.

---

## Contributing

Contributions are welcome. Please open issues or pull requests on GitHub. If you need support, use GitHub Issues to contact us.
