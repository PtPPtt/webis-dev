"""数据源获取 Agent：用 LLM 做工具路由，并在主线程同步执行工具。"""

# 用法：
#   python -m crawler.agent "帮我搜索 llm 相关文件" --limit 20
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None
    load_dotenv = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "缺少 LangChain 相关依赖，请先安装：`pip install langchain langchain-openai`。"
    ) from exc

if __package__ is None:
    # 允许直接脚本运行：python crawler/agent.py ...
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from ddg_scrapy_tool import DuckDuckGoScrapyTool  # type: ignore  # noqa: E402
    from tool_base import BaseTool, ToolResult  # type: ignore  # noqa: E402
else:
    from .ddg_scrapy_tool import DuckDuckGoScrapyTool  # type: ignore
    from .tool_base import BaseTool, ToolResult  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_env():
    """Load .env or .env.local if python-dotenv is available."""
    if not find_dotenv or not load_dotenv:
        return
    project_root = pathlib.Path(__file__).resolve().parents[1]
    for fname in (".env", ".env.local"):
        direct = project_root / fname
        if direct.exists():
            load_dotenv(str(direct), override=False)
            logger.info("Loaded environment from %s", direct)
            return
    for fname in (".env", ".env.local"):
        path = find_dotenv(fname, usecwd=True)
        if path:
            load_dotenv(path, override=False)
            logger.info("Loaded environment from %s", path)
            return


class LangChainDataSourceAgent:
    """
    数据源获取 Agent：
    - 用 LLM 在「已注册 tools」中选择最合适的一个
    - 然后在主线程同步调用 tool.run(...)（避免 Scrapy/Twisted 在线程池里跑导致 signal 报错）
    """

    def __init__(self, llm, tools: Optional[List[BaseTool]] = None, verbose: bool = False):
        if tools is None:
            tools = []
        self.llm = llm
        self.verbose = verbose
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.last_choice: Optional[dict] = None

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def available_tools(self) -> List[str]:
        return list(self.tools.keys())

    def run(self, task: str, **kwargs) -> ToolResult:
        logger.info("Agent received task: %s", task)
        if not self.tools:
            return ToolResult(name="agent", success=False, error="未注册任何 tools，无法执行任务。")

        tool_name, tool_task, tool_kwargs = self._choose_tool(task=task, **kwargs)
        self.last_choice = {"tool_name": tool_name, "tool_task": tool_task, "tool_kwargs": tool_kwargs}

        tool = self.tools.get(tool_name)
        if tool is None:
            fallback = next(iter(self.tools.values()))
            logger.warning("LLM chose unknown tool '%s', fallback to '%s'", tool_name, fallback.name)
            tool = fallback
            tool_name = tool.name

        merged_kwargs = dict(tool_kwargs)
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        try:
            return tool.run(task=tool_task, **merged_kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                name=tool_name,
                success=False,
                error=str(exc),
                meta={"task": task, "tool_task": tool_task, "tool_kwargs": merged_kwargs},
            )

    def _choose_tool(self, task: str, **kwargs) -> Tuple[str, str, dict]:
        """
        用一次 LLM 调用做路由：
        返回 (tool_name, tool_task, tool_kwargs)
        """
        tools_spec = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        system = (
            "你是数据源获取助手。你只能从给定的工具列表中选择一个最合适的工具来执行用户任务。\n"
            "你必须输出一个 JSON 对象，不能输出其他任何文字。\n"
            "JSON 格式：\n"
            '{\n  "tool_name": "xxx",\n  "tool_task": "给工具的更具体任务/搜索关键词",\n  "tool_kwargs": {\n    "limit": 5\n  }\n}\n'
            "规则：\n"
            "1) tool_name 必须是工具列表中的某一个 name。\n"
            "2) tool_task 必须可直接传给 tool.run(task=...)。\n"
            "3) tool_kwargs 只放必要参数；未给的参数由调用方默认值决定。\n"
            "工具列表：\n"
            f"{tools_spec}\n"
        )
        user = f"用户任务：{task}"

        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = getattr(resp, "content", "") or ""
        data = self._safe_json_loads(content) or {}

        tool_name = str(data.get("tool_name") or "").strip()
        tool_task = str(data.get("tool_task") or task).strip() or task
        tool_kwargs = data.get("tool_kwargs") if isinstance(data.get("tool_kwargs"), dict) else {}

        if "limit" in kwargs and "limit" not in tool_kwargs:
            tool_kwargs["limit"] = kwargs["limit"]

        if not tool_name:
            tool_name = next(iter(self.tools.keys()))

        return tool_name, tool_task, tool_kwargs

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[dict]:
        text = (text or "").strip()
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        candidate = fenced.group(1) if fenced else text
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            pass
        m = re.search(r"(\{.*\})", candidate, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            return None


def cli():
    parser = argparse.ArgumentParser(description="Data-source agent demo.")
    parser.add_argument("task", help="Natural-language task/keyword to fetch data for.")
    parser.add_argument("--limit", type=int, default=5, help="Max results for crawler tools.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.2", help="LLM model name (SiliconFlow).")
    parser.add_argument("--api-key", type=str, default=None, help="SiliconFlow API key (env SILICONFLOW_API_KEY).")
    args = parser.parse_args()

    _load_env()

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("需要安装 langchain-openai 才能在 CLI 演示中创建 LLM：`pip install langchain-openai`") from exc

    api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 SiliconFlow API Key，请设置环境变量 SILICONFLOW_API_KEY 或使用 --api-key 传入。")

    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
    )
    ddg_tool = DuckDuckGoScrapyTool()
    agent = LangChainDataSourceAgent(llm=llm, tools=[ddg_tool], verbose=True)

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
