"""Standalone demo for structuring module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extract_agent import StructureExtractionAgent
from .llm import get_default_llm
from .prompt_agent import PromptBuilderAgent


def read_texts(inputs):
    texts = []
    for p in inputs:
        path = Path(p)
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def main():
    parser = argparse.ArgumentParser(description="Structuring module demo.")
    parser.add_argument("goal", help="用户的结构化目标（自然语言）")
    parser.add_argument("--inputs", nargs="+", required=True, help="输入文本文件路径列表")
    parser.add_argument("--format", default="json", choices=["json", "markdown", "table"], help="输出格式")
    parser.add_argument("--schema", default=None, help="可选的 schema JSON 文件路径")
    parser.add_argument("--fewshots", default=None, help="可选的 few-shot JSON 文件路径")
    parser.add_argument("--limit", type=int, default=3, help="仅用于 demo，截取前 N 个文本做 prompt 构造")
    args = parser.parse_args()

    llm = get_default_llm()
    texts = read_texts(args.inputs)

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8")) if args.schema else None
    fewshots = json.loads(Path(args.fewshots).read_text(encoding="utf-8")) if args.fewshots else None

    prompt_agent = PromptBuilderAgent(llm)
    extraction_agent = StructureExtractionAgent(llm)

    prompt = prompt_agent.build(
        goal=args.goal,
        texts=texts[: args.limit],
        output_format=args.format,
        schema=schema,
        few_shots=fewshots,
    )
    print("\n===== Generated Extraction Prompt =====\n")
    print(prompt)

    result = extraction_agent.extract(prompt=prompt, text=texts, output_format=args.format)
    print("\n===== Extraction Result =====\n")

    if result.output_format == "json" and result.parsed is not None:
        print(json.dumps(result.parsed, ensure_ascii=False, indent=2))
    else:
        print(result.raw)


if __name__ == "__main__":
    main()

