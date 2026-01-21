"""
Improved Report Generation Demo

This demo showcases the refined report generation with:
- Concise information extraction (no verbosity)
- Source attribution for every piece of information
- LLM-powered synthesis with structure guidance
- Top-N document focus (not all documents)

Features:
- Explicit source tracking: 【来源：文档名】
- Minimal redundancy
- LLM-guided extraction through precise prompts
- Multiple demonstration scenarios

Run:
  python RAG/demo_improved_report.py
"""
from __future__ import annotations

import os
from pathlib import Path
from rag_pipeline import RAGPipeline
from rag_tasks import TaskPipeline, ReportGenerationTask


def get_openai_llm():
    """Initialize OpenAI LLM client"""
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, running without LLM")
            return None
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key,
        )
        print("✓ Using OpenAI GPT-4o-mini")
        return llm
    except ImportError:
        print("⚠️  langchain-openai not installed")
        return None


def demo_1_concise_report():
    """Demo 1: Generate concise report with source attribution"""
    print("\n" + "=" * 70)
    print("DEMO 1: 精炼报告 - 带来源标注")
    print("=" * 70)
    
    llm = get_openai_llm()
    
    rag_pipeline = RAGPipeline(
        min_doc_threshold=4,
        min_score_threshold=0.4,
    )
    
    query = "tell me the recent development of artificial intelligence"
    print(f"\n📌 查询: {query}\n")
    
    # Get context
    print("第1步：检索相关文档...")
    context = rag_pipeline.get_retrieval_context(query, auto_fetch_webis=True, top_k=5)
    print(f"✓ 检索到 {context['metadata']['retrieval_count']} 篇文档\n")
    
    # Generate report
    print("第2步：生成精炼报告（含来源）...")
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(ReportGenerationTask(
        llm=llm,
        include_raw_data=False,  # Only concise info
        output_format="markdown"
    ))
    
    result = task_pipeline.execute(context)
    
    if result['task_results'][0]['success']:
        report_path = result['task_results'][0]['output_path']
        print(f"✓ 报告已生成")
        print(f"  文件: {Path(report_path).name}")
        print(f"  内容长度: {result['task_results'][0]['stats']['content_length']} 字符")
        
        # Show report preview
        print(f"\n📄 报告预览:")
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            for line in lines[:30]:  # Show first 30 lines
                print(line)
            if len(lines) > 30:
                print(f"... (还有 {len(lines) - 30} 行)")


def demo_2_source_tracking():
    """Demo 2: Demonstrate source tracking in all sections"""
    print("\n" + "=" * 70)
    print("DEMO 2: 来源追踪 - 每条信息都标注出处")
    print("=" * 70)
    
    llm = get_openai_llm()
    
    rag_pipeline = RAGPipeline(
        min_doc_threshold=1,
        min_score_threshold=0.2,
    )
    
    query = "云计算的发展趋势"
    print(f"\n📌 查询: {query}\n")
    
    # Get context
    print("检索中...")
    context = rag_pipeline.get_retrieval_context(query, auto_fetch_webis=True)
    print(f"✓ 检索到 {context['metadata']['retrieval_count']} 篇文档\n")
    
    # Generate report with source tracking
    print("生成带来源追踪的报告...")
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(ReportGenerationTask(
        llm=llm,
        include_raw_data=False,
        output_format="markdown"
    ))
    
    result = task_pipeline.execute(context)
    
    if result['task_results'][0]['success']:
        report_path = result['task_results'][0]['output_path']
        
        # Read and display sections with source tracking
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("\n提取的关键发现（带来源标注）:")
        print("-" * 70)
        
        in_findings = False
        for line in content.split('\n'):
            if "## 关键发现" in line:
                in_findings = True
                continue
            if in_findings and line.startswith("## "):
                break
            if in_findings and line.strip().startswith("- 【来源"):
                print(line)


def demo_3_comparison_with_without_llm():
    """Demo 3: Compare report with and without LLM"""
    print("\n" + "=" * 70)
    print("DEMO 3: 对比 - 有LLM vs 无LLM")
    print("=" * 70)
    
    llm = get_openai_llm()
    
    rag_pipeline = RAGPipeline(
        min_doc_threshold=5,
        min_score_threshold=0.4,
    )
    
    query = "区块链技术应用"
    print(f"\n📌 查询: {query}\n")
    
    context = rag_pipeline.get_retrieval_context(query, auto_fetch_webis=True, top_k=5)
    
    # Report without LLM
    print("生成报告 (不使用LLM)...")
    task_pipeline_no_llm = TaskPipeline()
    task_pipeline_no_llm.add_task(ReportGenerationTask(
        llm=None,
        include_raw_data=False,
        output_format="markdown"
    ))
    result_no_llm = task_pipeline_no_llm.execute(context)
    
    if result_no_llm['task_results'][0]['success']:
        size_no_llm = result_no_llm['task_results'][0]['stats']['content_length']
        print(f"✓ 无LLM报告: {size_no_llm} 字符")
    
    # Report with LLM
    if llm:
        print("生成报告 (使用LLM)...")
        task_pipeline_llm = TaskPipeline()
        task_pipeline_llm.add_task(ReportGenerationTask(
            llm=llm,
            include_raw_data=False,
            output_format="markdown"
        ))
        result_llm = task_pipeline_llm.execute(context)
        
        if result_llm['task_results'][0]['success']:
            size_llm = result_llm['task_results'][0]['stats']['content_length']
            print(f"✓ 使用LLM报告: {size_llm} 字符")
            print(f"\n对比:")
            print(f"  - 有LLM: 更精炼、更结构化")
            print(f"  - 无LLM: 简单提取、可能有冗余")


def demo_4_batch_generation():
    """Demo 4: Batch generation with source tracking"""
    print("\n" + "=" * 70)
    print("DEMO 4: 批量生成 - 多查询批处理")
    print("=" * 70)
    
    llm = get_openai_llm()
    
    rag_pipeline = RAGPipeline(
        min_doc_threshold=1,
        min_score_threshold=0.2,
    )
    
    queries = [
        "5G技术发展",
        "量子计算",
        "边缘计算",
    ]
    
    print("\n生成多个报告...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. 处理: {query}")
        
        try:
            context = rag_pipeline.get_retrieval_context(query, auto_fetch_webis=True)
            
            task_pipeline = TaskPipeline()
            task_pipeline.add_task(ReportGenerationTask(
                llm=llm,
                include_raw_data=False,
                output_format="markdown"
            ))
            
            result = task_pipeline.execute(context)
            
            if result['task_results'][0]['success']:
                report_path = result['task_results'][0]['output_path']
                size = result['task_results'][0]['stats']['content_length']
                print(f"   ✓ 生成: {Path(report_path).name} ({size} 字符)")
            else:
                print(f"   ✗ 失败: {result['task_results'][0].get('error')}")
        
        except Exception as e:
            print(f"   ✗ 错误: {e}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("精炼报告生成演示 - 带来源标注")
    print("=" * 70)
    print("\n展示改进的报告生成功能：")
    print("  ✓ 精炼的信息提取")
    print("  ✓ 来源标注：【来源：文档名】")
    print("  ✓ 无冗余")
    print("  ✓ LLM增强的结构化")
    print()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  提示: 设置 OPENAI_API_KEY 可获得更好的LLM驱动的报告")
        print("   export OPENAI_API_KEY='sk-...'\n")
    
    try:
        demo_1_concise_report()
        # demo_2_source_tracking()
        # demo_3_comparison_with_without_llm()
        # demo_4_batch_generation()
        
        print("\n" + "=" * 70)
        print("✓ 所有演示完成")
        print("=" * 70)
        print("\n📁 报告已保存至 ./data/ 目录")
        print("\n💡 改进要点:")
        print("  - 信息精炼，无冗余")
        print("  - 每条信息都有来源标注")
        print("  - LLM生成的提示词指导结构化输出")
        print("  - 支持多种输出格式（Markdown/PDF）")
        print()
        
    except Exception as e:
        print(f"\n❌ 演示执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
