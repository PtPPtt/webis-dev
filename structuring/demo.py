"""Standalone demo for structuring module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_agent import StructureExtractionAgent
from llm import get_default_llm
from prompt_agent import PromptBuilderAgent
from validation.manager import create_validator, ValidationManager  # 新增


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

    # 新增验证参数
    parser.add_argument("--validate", action="store_true", help="启用结果验证")
    parser.add_argument("--validation-config", default=None,
                        help="验证配置文件路径 (JSON)")

    args = parser.parse_args()

    llm = get_default_llm()
    texts = read_texts(args.inputs)

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8")) if args.schema else None
    fewshots = json.loads(Path(args.fewshots).read_text(encoding="utf-8")) if args.fewshots else None

    prompt_agent = PromptBuilderAgent(llm)
    extraction_agent = StructureExtractionAgent(llm)

    validator_config = {}
    # 如果指定了配置文件，加载其内容
    if args.validation_config:
        try:
            with open(args.validation_config, 'r', encoding='utf-8') as f:
                validator_config = json.load(f)
                print("使用配置文件：" + args.validation_config)
        except FileNotFoundError:
            print(f"警告：验证配置文件 '{args.validation_config}' 未找到，使用默认配置")
        except json.JSONDecodeError as e:
            print(f"警告：验证配置文件 '{args.validation_config}' JSON格式错误: {e}，使用默认配置")
        except Exception as e:
            print(f"警告：读取验证配置文件时出错: {e}，使用默认配置")

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
        result_text = json.dumps(result.parsed, ensure_ascii=False)
    else:
        print(result.raw)
        result_text = result.raw

    """ ========== 验证步骤 ========== """
    if args.validate and result.success:
        print("\n" + "=" * 60)
        print("启动智能验证与优化循环")
        print("=" * 60)

        # 创建增强的验证管理器
        from validation.manager import EnhancedValidationManager
        enhanced_validator = EnhancedValidationManager(llm, validator_config)

        # 运行自回归优化循环
        optimization_result = enhanced_validator.run_validation_loop(
            initial_prompt=prompt,  # 初始抽取Prompt
            initial_result=result_text,  # 初始提取结果
            original_text=texts,  # 原始文本
            original_goal=args.goal,  # 用户原始目标
            max_retries=2,  # 最大优化次数
            prompt_agent=prompt_agent,  # 传入PromptBuilderAgent
            extraction_agent=extraction_agent,
            output_format=args.format

        )

        # 输出最终结果
        print(f"\n最终质量得分: {optimization_result['final_score']:.2f}/5")
        print(f"优化轮次: {optimization_result['optimization_count']}")

        if optimization_result['passed']:
            print("✅ 经过优化，结果质量已达标！")
        else:
            print("⚠️  经过优化，结果质量仍有提升空间。")

        #  保存优化历史
        serializable_history = convert_to_serializable(optimization_result["history"])

        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)

        print("优化历史已保存到: output.json")


def convert_to_serializable(obj):
    """递归地将对象转换为可JSON序列化的格式"""
    if hasattr(obj, '__dict__'):
        # 如果是对象，转换为字典
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # 排除私有属性
                result[key] = convert_to_serializable(value)
        return result
    elif isinstance(obj, (list, tuple)):
        # 如果是列表或元组，递归处理每个元素
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # 如果是字典，递归处理每个值
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        # 基本数据类型直接返回
        return obj

if __name__ == "__main__":
    main()

