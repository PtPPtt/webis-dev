"""PromptBuilderAgent: build extraction prompts from user goals + texts."""

# 并没有调用专有的agent去生成专属于数据集的prompt，demo阶段只是一个固定的prompt格式

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
        # 新增：构建优化的链
        self._refine_chain = self._build_refine_chain()

    def build(self, goal: str, texts: List[str], output_format: str = "json", schema=None, few_shots=None) -> str:
        payload = PromptBuildInput(
            goal=goal,
            texts=texts,
            output_format=output_format,
            schema=schema,
            few_shots=few_shots or [],
        )
        return self._chain.invoke(payload.__dict__)

    def refine(self, original_prompt: str, feedback: str) -> str:
        """
        基于质量反馈优化抽取Prompt。

        Args:
            original_prompt: 原始抽取Prompt
            feedback: 质量评估反馈及优化建议（来自FeedbackGenerator）
            validation_scores: 各维度验证分数（供参考，格式与validation_result.rater_results一致）

        Returns:
            优化后的抽取Prompt
        """
        # 直接使用优化链处理
        refine_input = {
            "original_prompt": original_prompt,
            "feedback": feedback,
        }
        return self._refine_chain.invoke(refine_input)

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
                    "5) 最终只输出『抽取 Prompt』本身，不要额外解释。\n"
                    "重要约束：\n"
                    "1、最终输出的 prompt 字符串里不得包含任何占位符，若要有，则用'()'代替。\n"
                    ,
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

    def _build_refine_chain(self):
        """构建Prompt优化链"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个Prompt优化助手。请严格根据提供的优化反馈，直接对原始抽取Prompt进行优化。\n"
                    "优化原则：\n"
                    "1. 保持原始Prompt的核心任务不变。\n"
                    "2. 只根据反馈中的具体问题和建议进行修改。\n"
                    "3. 让指令更明确，格式更规范。\n"
                    "4. 输出完整的优化后Prompt，不添加解释。\n"
                    "重要约束：\n"
                    "1、最终输出的 prompt 字符串里不得包含任何占位符，若要有，则用'()'代替。\n",
                ),
                (
                    "human",
                    "【原始Prompt】\n"
                    "{original_prompt}\n\n"
                    "【优化反馈与建议】\n"
                    "{feedback}\n\n"
                    "请基于以上反馈，输出优化后的完整抽取Prompt："
                ),
            ]
        )
        return prompt | self.llm | StrOutputParser()