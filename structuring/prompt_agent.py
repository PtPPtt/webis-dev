"""PromptBuilderAgent: build extraction prompts from user goals + texts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class PromptBuildInput:
    goal: str
    texts: List[str]
    output_format: str = "json"  # json | markdown | table
    schema: Optional[Dict[str, Any]] = None  # {"fields": [{"name":..., "type":..., "desc":...}, ...]}
    few_shots: List[Dict[str, Any]] = field(default_factory=list)


class PromptBuilderAgent:
    """
    根据用户结构化目标 + 输入文本生成抽取 Prompt。
    - 可传 schema 指定字段模板
    - 可传 few_shots 提供 few-shot 示例
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = self._build_chain()

    def build(self, goal: str, texts: List[str], output_format: str = "json", schema=None, few_shots=None) -> str:
        payload = PromptBuildInput(
            goal=goal,
            texts=texts,
            output_format=output_format,
            schema=schema,
            few_shots=few_shots or [],
        )
        return self._chain.invoke(payload.__dict__)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 Prompt 构造助手。你的任务是根据用户的结构化目标和输入文本，"
                    "生成一个高质量的『结构抽取 Prompt』给下游抽取 Agent 使用。\n"
                    "要求：\n"
                    "1) 明确抽取目标与字段。\n"
                    "2) 指定输出格式（json/markdown/table）。\n"
                    "3) 如果给了 schema，必须严格按 schema 字段定义抽取；"
                    "如果没有 schema，则根据 goal 推断字段。\n"
                    "4) 如果给了 few_shots，请将其整理为 few-shot 示例并嵌入 Prompt。\n"
                    "5) 最终只输出『抽取 Prompt』本身，不要额外解释。\n",
                ),
                (
                    "human",
                    "用户结构化目标：\n{goal}\n\n"
                    "期望输出格式：{output_format}\n\n"
                    "字段模板 schema（可能为空）：\n{schema}\n\n"
                    "Few-shot 示例（可能为空）：\n{few_shots}\n\n"
                    "输入文本（节选，供你推断字段/写示例）：\n"
                    "{texts}\n\n"
                    "请生成下游可直接使用的结构抽取 Prompt。",
                ),
            ]
        )
        return prompt | self.llm | StrOutputParser()

