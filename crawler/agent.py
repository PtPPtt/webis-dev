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
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))

    from ddg_scrapy_tool import DuckDuckGoScrapyTool
    from tool_base import BaseTool, ToolResult
    from gnews_tool import GNewsTool
    from semantic_scholar import SemanticScholarTool
    from github_api_tools import GitHubSearchTool
    from hn_tool import HackerNewsTool
else:
    from .ddg_scrapy_tool import DuckDuckGoScrapyTool
    from .tool_base import BaseTool, ToolResult
    from .gnews_tool import GNewsTool
    from .semantic_scholar import SemanticScholarTool
    from .github_api_tools import GitHubSearchTool
    from .hn_tool import HackerNewsTool


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_env():
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
        limit: int = 5,
        verbose: bool = False,
    ):
        self.llm = llm
        self.verbose = verbose
        self.limit = limit

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

    def run(self, task: str) -> ToolResult:
        logger.info("Agent received task: %s", task)
        self.last_result = None

        raw_output = self.agent.run(task)

        if self.last_result:
            return self.last_result

        return ToolResult(
            name="langchain_agent",
            success=False,
            error="Agent finished without tool result.",
            meta={"raw_output": raw_output},
        )

    def _wrap_tool_for_langchain(self, tool: BaseTool) -> Tool:
        """
        LangChain agent ONLY passes a single string argument to tools.
        """

        def _run(input: str):
            result = tool.run(task=input, limit=self.limit)
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
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": (
                    "你是一个数据采集 Agent，不是聊天助手。\n"
                    "你必须使用工具获取真实数据。\n"
                    "如果某个工具无法获取结果，必须明确失败原因，禁止改用其他工具替代。\n"
                    "禁止生成 JSON 文件或空结果作为最终答案。\n"
                    "最终结果只能来自工具返回的内容。"
                )
            },
        )


def cli():
    parser = argparse.ArgumentParser(description="Data-source agent demo.")
    parser.add_argument("task", help="Natural-language task to fetch data for.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs")

    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _load_env()

    from langchain_openai import ChatOpenAI

    api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 SiliconFlow API Key，请设置 SILICONFLOW_API_KEY")

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
            GNewsTool(output_dir=str(output_dir)),
            SemanticScholarTool(output_dir=str(output_dir)),
            GitHubSearchTool(output_dir=str(output_dir)),
            HackerNewsTool(output_dir=str(output_dir)),
        ],
        limit=args.limit,
        verbose=True,
    )

    result = agent.run(task=args.task)

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
