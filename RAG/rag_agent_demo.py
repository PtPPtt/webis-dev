"""
Simple demo: Shows how to use `WebisRAGAgent` to get enhanced prompt and context when LLM is not configured.

Run:
  python -m structuring.rag_agent_demo
"""
from __future__ import annotations

from rag_agent import WebisRAGAgent

def main():
    agent = WebisRAGAgent(llm=None)

    # First query will trigger fetch (empty RAG or low score)
    r1 = agent.handle_query("Tell me the recent news of china ecnomic", force_refresh=False)
    print("--- First query ---")
    print("Used webis:", r1["used_webis"])
    print("Retrieved docs:", r1["retrieved_documents"])
    # print("Enhanced prompt snippet:\n", r1["enhanced_prompt"])
    # 保存到当前目录的 enhanced_prompt.txt 文件中
    with open("enhanced_prompt.txt", "w", encoding="utf-8") as f:
        f.write(r1["enhanced_prompt"])


    # # Second same query should retrieve from RAG, usually no longer trigger fetch
    # r2 = agent.handle_query("What are the recent advancements in AI chips?", force_refresh=False)
    # print("\n--- Second query ---")
    # print("Used webis:", r2["used_webis"])
    # print("Retrieved docs:", r2["retrieved_documents"])


if __name__ == "__main__":
    main()
