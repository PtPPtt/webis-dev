"""
RAG System Comprehensive Demo

Demonstrates:
1. RAGPipeline - Independent RAG module for document processing and retrieval
2. RAG Tasks - Document extraction, summarization, prompt enhancement
3. TaskPipeline - Orchestrate multiple tasks in sequence

Recommended approach: Use RAGPipeline + TaskPipeline directly
(WebisRAGAgent is deprecated, kept for backward compatibility)

Run:
  python RAG/rag_agent_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path
from rag_pipeline import RAGPipeline
from rag_tasks import (
    TaskPipeline,
    PromptEnhancementTask,
    DocumentExtractionTask,
    SummaryTask
)


def demo_1_basic_pipeline():
    """Demo 1: Basic usage - RAGPipeline + PromptEnhancementTask"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic RAG Pipeline with PromptEnhancementTask")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        rag_store_path="./data/rag_store.json",
        embedding_model_type="gemma",
        top_k=3,
    )
    
    # Initialize task pipeline
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(PromptEnhancementTask(llm_agent=None))
    
    query = "Tell me about recent AI advancements in 2024"
    print(f"\n📌 Query: {query}\n")
    
    # Step 1: Get RAG context
    print("Step 1: Retrieving from RAG...")
    rag_context = rag_pipeline.get_retrieval_context(query, top_k=3)
    print(f"✓ Documents retrieved: {rag_context['metadata']['retrieval_count']}")
    
    # Step 2: Execute task pipeline
    print("Step 2: Executing task pipeline...")
    result = task_pipeline.execute(rag_context)
    
    # Save enhanced prompt
    if result['task_results']:
        task_result = result['task_results'][0]
        if task_result['success']:
            enhanced_prompt = task_result.get('enhanced_prompt', '')
            output_file = Path("demo1_enhanced_prompt.txt")
            output_file.write_text(enhanced_prompt, encoding="utf-8")
            print(f"✓ Enhanced prompt saved to {output_file}")
            print(f"\nEnhanced prompt preview:\n{enhanced_prompt[:300]}...\n")


def demo_2_custom_tasks():
    """Demo 2: Custom task pipeline with multiple tasks"""
    print("\n" + "=" * 70)
    print("DEMO 2: Multiple Tasks in Sequence")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(top_k=5)
    
    # Create task pipeline with multiple tasks
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(DocumentExtractionTask())
    task_pipeline.add_task(SummaryTask(max_summary_length=300))
    task_pipeline.add_task(PromptEnhancementTask(llm_agent=None))
    
    query = "What are the recent developments in quantum computing?"
    print(f"\n📌 Query: {query}\n")
    
    # Get RAG context
    print("Step 1: Retrieving from RAG...")
    rag_context = rag_pipeline.get_retrieval_context(query, top_k=5)
    print(f"✓ Documents retrieved: {rag_context['metadata']['retrieval_count']}")
    
    # Execute task pipeline
    print("Step 2: Executing task pipeline...")
    result = task_pipeline.execute(rag_context)
    
    # Display results from each task
    for i, task_result in enumerate(result['task_results'], 1):
        task_name = task_result.get('task_name', 'Unknown')
        print(f"\n--- Task {i}: {task_name} ---")
        
        if not task_result.get('success'):
            print(f"  Error: {task_result.get('error')}")
            continue
        
        if task_name == "document_extraction":
            print(f"  Document count: {task_result.get('document_count')}")
            for doc in task_result.get('extracted_documents', [])[:2]:
                print(f"    - {doc.get('source')}: {doc.get('content')[:80]}...")
        
        elif task_name == "summarization":
            print(f"  Source count: {task_result.get('source_count')}")
            summary = task_result.get('summary', '')
            print(f"  Summary: {summary[:150]}...")
            print(f"  Key points: {task_result.get('key_points')}")
        
        elif task_name == "prompt_enhancement":
            enhanced = task_result.get('enhanced_prompt', '')
            print(f"  Enhanced prompt length: {len(enhanced)}")
            print(f"  Preview: {enhanced[:150]}...")
    
    # Save results
    output_file = Path("demo2_task_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Results saved to {output_file}")


def demo_3_rag_pipeline_only():
    """Demo 3: Use RAGPipeline independently (without agent)"""
    print("\n" + "=" * 70)
    print("DEMO 3: RAGPipeline Independent Usage")
    print("=" * 70)
    
    # Create RAG pipeline directly
    pipeline = RAGPipeline(
        chunk_strategy="sliding_window",
        chunk_size=512,
        embedding_model_type="gemma",
        top_k=3
    )
    
    query = "What is the impact of large language models?"
    print(f"\n📌 Query: {query}\n")
    
    # Retrieve documents
    print("Retrieving documents from RAG...")
    retrieval_result = pipeline.retrieve(query, top_k=3)
    
    print(f"✓ Retrieved {retrieval_result['total_retrieved']} documents")
    print(f"✓ Top-K: {retrieval_result['top_k']}")
    
    if retrieval_result['documents']:
        print("\nRetrieved documents:")
        for i, (doc, score) in enumerate(zip(retrieval_result['documents'], retrieval_result['scores']), 1):
            print(f"  {i}. {doc['source']} (relevance: {score:.3f})")
            print(f"     {doc['content'][:100]}...")
    
    # Get full context for downstream tasks
    print("\nGetting comprehensive context...")
    context = pipeline.get_retrieval_context(query, top_k=3)
    
    print(f"Context structure:")
    print(f"  - Query: {context['query']}")
    print(f"  - Retrieved documents: {context['metadata']['retrieval_count']}")
    print(f"  - Structured data: {len(context['structured_data'])} items")
    print(f"  - Context text length: {len(context['context_text'])} chars")
    
    # Display stats
    print("\n📊 RAG Pipeline Statistics:")
    stats = pipeline.get_stats()
    print(f"  - Total documents: {stats['total_documents']}")
    print(f"  - Indexed: {stats['indexed']}")
    print(f"  - Store path: {stats['store_path']}")


def demo_4_query_variations():
    """Demo 4: Test multiple queries"""
    print("\n" + "=" * 70)
    print("DEMO 4: Multiple Query Testing")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Initialize task pipeline
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(PromptEnhancementTask(llm_agent=None))
    
    queries = [
        "Tell me about recent AI breakthroughs",
        "What are the latest developments in AI?",
        "Recent advancements in artificial intelligence",
    ]
    
    results = {}
    for i, query in enumerate(queries, 1):
        print(f"\n📌 Query {i}: {query}\n")
        
        # Get RAG context
        rag_context = rag_pipeline.get_retrieval_context(query)
        print(f"✓ Retrieved: {rag_context['metadata']['retrieval_count']} docs")
        print(f"✓ Scores: {rag_context['scores'][:3]}")
        
        # Execute tasks
        task_result = task_pipeline.execute(rag_context)
        
        results[query] = {
            "retrieved_count": rag_context['metadata']['retrieval_count'],
            "scores": rag_context['scores'],
            "task_success": task_result['task_results'][0]['success'] if task_result['task_results'] else False,
        }
    
    # Save results
    output_file = Path("demo4_query_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Query results saved to {output_file}")


def demo_5_task_chaining():
    """Demo 5: Complex task chaining"""
    print("\n" + "=" * 70)
    print("DEMO 5: Advanced Task Chaining")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(top_k=5)
    
    # Create task pipeline with specific order
    task_pipeline = TaskPipeline()
    task_pipeline.add_task(DocumentExtractionTask())
    task_pipeline.add_task(SummaryTask(max_summary_length=500))
    task_pipeline.add_task(PromptEnhancementTask(llm_agent=None))
    
    query = "Recent developments in cloud computing"
    print(f"\n📌 Query: {query}\n")
    
    # Step 1: Get RAG context
    print("Step 1: Retrieving context from RAG...")
    rag_context = rag_pipeline.get_retrieval_context(query, top_k=5)
    print(f"✓ Retrieved {rag_context['metadata']['retrieval_count']} documents")
    
    # Step 2: Execute task pipeline
    print("\nStep 2: Executing task pipeline...")
    task_result = task_pipeline.execute(rag_context)
    
    print(f"✓ Task execution success: {task_result['success']}")
    print(f"✓ Tasks executed: {len(task_result['task_results'])}")
    
    for i, result in enumerate(task_result['task_results'], 1):
        print(f"\n  Task {i}: {result.get('task_name')}")
        print(f"    - Success: {result.get('success')}")
        if result.get('error'):
            print(f"    - Error: {result.get('error')}")
    
    # Save full results
    output_file = Path("demo5_task_chaining_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task_result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("RAG SYSTEM COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("\n✓ Using RAGPipeline + TaskPipeline directly (recommended approach)")
    print("✓ WebisRAGAgent deprecated - kept for backward compatibility only\n")
    
    try:
        # Run demos
        demo_1_basic_pipeline()
        demo_2_custom_tasks()
        demo_3_rag_pipeline_only()
        demo_4_query_variations()
        demo_5_task_chaining()
        
        print("\n" + "=" * 70)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n📁 Output files generated:")
        print("  - demo1_enhanced_prompt.txt")
        print("  - demo2_task_results.json")
        print("  - demo4_query_results.json")
        print("  - demo5_task_chaining_results.json")
        print("\n💡 Summary:")
        print("  - RAGPipeline: Independent document retrieval")
        print("  - RAGTask: Pluggable task implementations")
        print("  - TaskPipeline: Orchestrates task execution")
        print("  - PromptEnhancementTask: Generate enhanced prompts")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
