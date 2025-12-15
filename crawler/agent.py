"""
Data-source agent built on LangChain to route tasks to tools.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None
    load_dotenv = None

# ===== LangChain imports (stable for 0.1.x) =====
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain_core.tools import Tool
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "缺少 LangChain 相关依赖，请确认安装："
        "`pip install langchain==0.1.20 langchain-openai==0.1.7`"
    ) from exc

# ===== Local tool imports =====
if not __package__:
    # 脚本方式运行：python agent.py
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))

    from ddg_scrapy_tool import DuckDuckGoScrapyTool
    from tool_base import BaseTool, ToolResult
    from gnews_tool import GNewsTool
    from semantic_scholar import SemanticScholarTool
    from github_api_tools import GitHubSearchTool
    from hn_tool import HackerNewsTool
else:
    # 包方式运行：python -m crawler.agent
    from .ddg_scrapy_tool import DuckDuckGoScrapyTool
    from .tool_base import BaseTool, ToolResult
    from .gnews_tool import GNewsTool
    from .semantic_scholar import SemanticScholarTool
    from .github_api_tools import GitHubSearchTool
    from .hn_tool import HackerNewsTool


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_env():
    """Load .env or .env.local if python-dotenv is available."""
    if not find_dotenv or not load_dotenv:
        return
    for fname in (".env.local", ".env"):
        path = find_dotenv(fname, usecwd=True)
        if path:
            load_dotenv(path, override=False)
            logger.info("Loaded environment from %s", path)
            break


class LangChainDataSourceAgent:
    """
    LangChain-based agent that uses an LLM to decide which tool to call.
    Compatible with langchain 0.1.x (initialize_agent).
    """

    def __init__(
        self,
        llm,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.verbose = verbose
        self.tools: Dict[str, BaseTool] = {}
        self.lc_tools: List[Tool] = []
        self.last_result: Optional[ToolResult] = None

        if tools:
            for tool in tools:
                self.register_tool(tool)

        self.agent = self._build_agent()

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        self.lc_tools.append(self._wrap_tool_for_langchain(tool))
        logger.info("Registered tool: %s", tool.name)

    def available_tools(self) -> List[str]:
        return list(self.tools.keys())

    def run(self, task: str, **kwargs) -> ToolResult:
        logger.info("Agent received task: %s", task)
        self.last_result = None

        _ = self.agent.run(task)

        if self.last_result:
            return self.last_result

        return ToolResult(
            name="langchain_agent",
            success=False,
            error="Agent finished without tool result.",
            meta={"raw_output": _},
        )

    def _wrap_tool_for_langchain(self, tool: BaseTool) -> Tool:
        """Wrap BaseTool as a LangChain Tool and capture execution result."""

        def _run(task: str, limit: int = 5, **kwargs):
            result = tool.run(task=task, limit=limit, **kwargs)
            self.last_result = result
            if result.success:
                return f"success; files={len(result.files)}"
            return f"error: {result.error}"

        return Tool(
            name=tool.name,
            description=tool.description,
            func=_run,
        )

    def _build_agent(self):
        return initialize_agent(
            tools=self.lc_tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
        )


def cli():
    parser = argparse.ArgumentParser(description="Data-source agent demo.")
    parser.add_argument("task", help="Natural-language task to fetch data for.")
    parser.add_argument("--limit", type=int, default=5, help="Max results for tools.")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3.2",
        help="LLM model name (SiliconFlow).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="SiliconFlow API key (env SILICONFLOW_API_KEY).",
    )
    args = parser.parse_args()

    _load_env()

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "需要安装 langchain-openai：`pip install langchain-openai==0.1.7`"
        ) from exc

    api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 SiliconFlow API Key，请设置环境变量 SILICONFLOW_API_KEY"
        )

    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
    )

    agent = LangChainDataSourceAgent(
        llm=llm,
        tools=[
            DuckDuckGoScrapyTool(),
            GNewsTool(),
            SemanticScholarTool(),
            GitHubSearchTool(),
            HackerNewsTool(),
        ],
        verbose=True,
    )

    result = agent.run(task=args.task, limit=args.limit)

    if result.success:
        print(f"✓ Agent finished using tool: {result.name}")
        print(f"Output dir: {result.output_dir}")
        print("Files:")
        for f in result.files:
            print(f" - {f}")
    else:
        print(f"✗ Agent failed: {result.error}")


if __name__ == "__main__":
    cli()
