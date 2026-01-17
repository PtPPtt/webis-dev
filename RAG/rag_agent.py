"""
Conditional Triggered RAG Agent Framework.

Features:
- Determine whether to call external webis pipeline to obtain prior knowledge based on existing RAG storage
- If needed, call the externally provided `webis_fetcher` to add clean text to RAG and build index
- Format the retrieved prior knowledge into prompt and call structured/answering Agent

See example usage at the bottom of the module `example_usage()`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import subprocess
import tempfile
from pathlib import Path
import json
import sys

from rag_tools import RAGManager

logger = logging.getLogger(__name__)


class WebisRAGAgent:
    """RAG Agent: Decide whether to trigger webis to obtain prior knowledge based on query and call Agent to answer."""

    def __init__(
        self,
        llm,
        rag_store_path: str = "./data/rag_store.json",
        top_k: int = 3,
        min_score_threshold: float = 0.30,
    ):
        """
        Args:
            llm: LLM instance (compatible with existing Prompt/Extract Agent)
            rag_store_path: RAG storage path
            webis_fetcher: Optional function, receives `query` returns list of {"content":..., "source":...}
            top_k: retrieve top-k
            min_score_threshold: If the similarity of the most relevant document is higher than this value, it is considered that RAG has covered this field, no need to call webis again
        """
        self.llm = llm
        self.rag_manager = RAGManager(rag_store_path=rag_store_path)
        # When llm is not provided, delay or skip the creation of Prompt/Extract Agent for local debugging
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold
        # Ensure loading existing local RAG storage (if exists)
        try:
            self.rag_manager._try_load()
        except Exception:
            pass

    def save_store(self) -> None:
        """Explicitly save RAG storage to disk (if index is built, save vectors/vectorizer)"""
        try:
            self.rag_manager.build_and_save()
        except Exception:
            try:
                self.rag_manager.rag.save()
            except Exception:
                logger.debug("Failed to save RAG store")

    def load_store(self) -> None:
        """Explicitly load RAG storage from disk"""
        try:
            self.rag_manager._try_load()
        except Exception:
            logger.debug("Failed to load RAG store")

    def should_fetch_webis(self, query: str) -> bool:
        """Determine whether to fetch more documents from webis based on current RAG content and threshold."""
        # Try to retrieve and get scores; if RAG is empty or retrieval fails, return True (need to call internal pipeline)
        try:
            res = self.rag_manager.retrieve_for_query(query, top_k=self.top_k, include_scores=True)
        except Exception:
            return True

        scores = res.get("scores") or []
        # If no retrieval results, trigger pipeline
        if not scores:
            print("need to fetch: RAG has no relevant documents\n")
            return True
        print("RAG retrieval scores:", scores, "\n")
        top_score = max(scores)
        logger.debug(f"Top RAG score for query '{query}': {top_score:.3f}")
        return top_score < self.min_score_threshold

    def fetch_and_add(self, query: str) -> List[str]:
        """Call webis_fetcher to get documents and add to RAG, return list of added doc_ids."""
        # Directly call internal pipeline and add results to RAG
        return self.run_pipeline_and_store(query)

    def _save_webis_documents(self, documents: List, query: str) -> None:
        """Save WebisDocument objects to JSON files for persistence and inspection.
        
        Args:
            documents: List of WebisDocument objects from pipeline
            query: The query that produced these documents
        """
        try:
            from datetime import datetime
            
            # Create output directory
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "webis_documents"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp for file naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_safe = "".join(c if c.isalnum() else "_" for c in query)[:50]
            output_file = output_dir / f"webis_docs_{query_safe}_{timestamp}.json"
            
            # Prepare documents for serialization using to_json method
            docs_data = []
            for doc in documents:
                # Use the to_json method if available, otherwise fallback to manual conversion
                if hasattr(doc, 'to_json'):
                    doc_dict = doc.to_json(include_embeddings=False)
                else:
                    doc_dict = {
                        "id": doc.id,
                        "content": doc.content[:1000] if doc.content else "",
                        "clean_content": doc.clean_content[:1000] if doc.clean_content else "",
                        "doc_type": doc.doc_type.value if hasattr(doc.doc_type, 'value') else str(doc.doc_type),
                        "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                        "title": doc.meta.title or "",
                        "url": doc.meta.url or "",
                        "source_plugin": doc.meta.source_plugin or "",
                    }
                docs_data.append(doc_dict)
            
            # Prepare output structure
            output_data = {
                "query": query,
                "timestamp": timestamp,
                "document_count": len(docs_data),
                "documents": docs_data,
            }
            
            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Saved {len(docs_data)} WebisDocuments to {output_file}\n")
            logger.info(f"Saved WebisDocuments to {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save WebisDocuments: {e}")

    def fetch_and_add(self, query: str) -> List[str]:
        """Call webis_fetcher to get documents and add to RAG, return list of added doc_ids."""
        # Directly call internal pipeline and add results to RAG
        return self.run_pipeline_and_store(query)

    def handle_query(
        self,
        query: str,
        extraction_goal: str = "Answer user query based on reference information",
        output_format: str = "markdown",
        force_refresh: bool = False,
        use_webis_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """Main entry: Receive query, trigger webis as needed, build enhanced prompt and call Agent to return result.

        Return structure: {
            "query":..., "used_webis":bool, "retrieved_documents": [...],
            "enhanced_prompt":..., "agent_result": { ... }
        }
        """
        used_webis = False

        print("Determining whether to use webis to obtain prior knowledge\n")

        if use_webis_if_needed and (force_refresh or self.should_fetch_webis(query)):
            logger.info("Fetching documents from webis for query...")
            print("Calling webis pipeline to obtain prior knowledge...\n")
            added = self.fetch_and_add(query)
            used_webis = bool(added)

       
        base_prompt = f"User query: {query}\nPlease provide an answer in the following format based on available reference information (if any): {extraction_goal}"

        print("Webis retrieval completed, building enhanced prompt...\n")
        # Use RAG to enhance prompt
        enhanced_prompt, context_data = self.rag_manager.enhance_agent_prompt(
            base_prompt=base_prompt, user_query=query, rag_top_k=self.top_k
        )

        # If LLM or extractor is not configured, do not call extractor, only return enhanced prompt and context for debugging/offline testing
        if not self.llm or not self.extraction_agent:
            print("LLM or extraction agent not configured, skipping agent call.\n")
            result_obj = {
                "query": query,
                "used_webis": used_webis,
                "retrieved_documents": [d.source for d in context_data.get("documents", [])],
                "enhanced_prompt": enhanced_prompt,
                "agent_result": {
                    "success": False,
                    "output_format": output_format,
                    "parsed": None,
                    "raw": None,
                    "error": "LLM not configured",
                },
            }
            # Save answer (placeholder info) to data/agent_answers_<ts>.json
            try:
                from datetime import datetime

                out_dir = Path(__file__).resolve().parents[1] / "data"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"agent_answer_{ts}.json"
                out_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                logger.debug("Failed to save agent answer to disk")

            return result_obj

        # Use enhanced prompt to call extraction/answering Agent (structured extraction Agent is reused to generate answer)
        agent_result = self.extraction_agent.extract(
            prompt=enhanced_prompt, text="", output_format=output_format
        )

        result_obj = {
            "query": query,
            "used_webis": used_webis,
            "retrieved_documents": [d.source for d in context_data.get("documents", [])],
            "enhanced_prompt": enhanced_prompt,
            "agent_result": {
                "success": agent_result.success,
                "output_format": agent_result.output_format,
                "parsed": agent_result.parsed,
                "raw": agent_result.raw,
                "error": agent_result.error,
            },
        }

        # Save answer to data/agent_answers_<ts>.json
        try:
            from datetime import datetime

            out_dir = Path(__file__).resolve().parents[1] / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"agent_answer_{ts}.json"
            out_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to save agent answer to disk")

        return result_obj

    def run_pipeline_and_store(self, query: str, limit: int = 3, workers: int = 2) -> List[str]:
        """Call the complete pipeline based on the `webis_pipeline` code in the project and store the results in RAG.

        Main steps (refer to webis_pipeline.py implementation):
        - Call crawler to get data (using DuckDuckGo or Baidu search)
        - Fetch HTML content from URLs
        - Clean HTML and extract clean text
        - Parse documents and apply AI denoising
        - Deduplicate documents
        - Extract structured data using LLM
        - Add cleaned text and structured results to RAG and build index

        Return the list of newly added doc_ids.
        """
        # Use new Webis Pipeline API instead of legacy webis_pipeline module.
        try:
            from webis.core.pipeline import Pipeline
        except Exception as e:
            logger.warning(f"Webis pipeline not available: {e}")
            return []

        # Ensure we have an LLM; try to reuse existing llm or initialize lightly
        llm = self.llm

        # Build a comprehensive pipeline: search -> fetch -> clean -> parse -> deduplicate -> extract
        try:
            pipe = Pipeline()

            # Register all required plugins
            try:
                from webis.plugins.sources import DuckDuckGoPlugin
                from webis.plugins.processors import (
                    HtmlFetcherPlugin, 
                    HtmlCleanerPlugin, 
                    DocumentParsePlugin
                )
                from webis.plugins.processors import ChunkingPlugin
                from webis.plugins.processors import EmbeddingPlugin
                from webis.plugins.extractors import LLMExtractorPlugin
                
                registry = pipe.registry
                
                # Register source plugins
                try:
                    registry.register_class(DuckDuckGoPlugin)
                except Exception:
                    pass
                
                # Register processor plugins
                try:
                    registry.register_class(HtmlFetcherPlugin)
                except Exception:
                    pass
                try:
                    registry.register_class(HtmlCleanerPlugin)
                except Exception:
                    pass
                try:
                    registry.register_class(DocumentParsePlugin)
                except Exception:
                    pass

                # Register extractor plugins
                try:
                    registry.register_class(LLMExtractorPlugin)
                except Exception:
                    pass
                
                try:
                    registry.register_class(EmbeddingPlugin)
                except Exception:
                    pass
                
                try:
                    registry.register_class(ChunkingPlugin)
                except Exception:
                    pass
                    
            except Exception as e:
                logger.warning(f"Some plugins could not be imported: {e}")

            # Build pipeline: source -> processors -> extractors
            pipe.add_source("duckduckgo", max_results=limit)
            pipe.add_processor("html_fetcher")
            pipe.add_processor("html_cleaner")
            pipe.add_processor("chunker")
            # pipe.add_processor("document_parser")  # Parse DOCX, TXT, MD with AI denoising
            # pipe.add_processor("deduplication")     # Remove duplicates
            pipe.add_processor("embedder")          # Generate embeddings for document chunks
            
            # Add extractor for structured data
            pipe.add_extractor(
                "llm_extractor",
                config={
                    "output_format": "json",
                    "schema": {
                        "title": "string",
                        "key_points": ["string"],
                        "entities": ["string"],
                        "topics": ["string"],
                        "summary": "string"
                    }
                }
            )

            print(f"Running enriched pipeline for query: '{query}'...")
            print("Pipeline stages: search -> fetch -> clean -> parse -> deduplicate -> embedding -> extract structured data\n")
    
            result = pipe.run(query)

            # print("result:", result, "\n")
            
            # Display Results
            print(f"\nFound {result.document_count} results:\n")
            for i, doc in enumerate(result.documents, 1):
                print(f"--- Result {i} ---")
                print(f"Title: {doc.meta.title}")
                print(f"URL: {doc.meta.url}")
                
                # Display cleaned content
                clean_content = doc.clean_content
                if clean_content:
                    print(f"Cleaned Content: {clean_content[:200]}...")
                
                print("-" * 50)

            texts = []
            documents = []
            doc_structured_data = {}

            # Collect structured results from extractors
            if getattr(result, "structured_results", None):
                try:
                    # Aggregate all structured results by source document
                    for sr in result.structured_results:
                        doc_ids = getattr(sr, "lineage", {}).source_doc_ids if hasattr(sr, "lineage") else []
                        sr_data = getattr(sr, "data", None)
                        if sr_data and doc_ids:
                            for doc_id in doc_ids:
                                if doc_id not in doc_structured_data:
                                    doc_structured_data[doc_id] = {}
                                # Merge structured data from different extractors
                                if isinstance(sr_data, dict):
                                    doc_structured_data[doc_id].update(sr_data)
                except Exception as e:
                    logger.warning(f"Failed to collect structured results: {e}")

            # Extract cleaned data and structured data from documents
            for doc in result.documents:
                # Prefer cleaned content from document_parser or html_cleaner
                clean_content = getattr(doc, "clean_content", None) or getattr(doc, "content", None) or ""
                source = getattr(doc.meta, "url", None) or getattr(doc.meta, "title", None) or getattr(doc.meta, "source_plugin", None) or f"webis_{doc.id}"
                
                # Get structured data for this document
                doc_structured = doc_structured_data.get(doc.id, None)
                
                # Extract embeddings from chunks if available
                embeddings = []
                chunks_content = []
                if hasattr(doc, "chunks") and doc.chunks:
                    for chunk in doc.chunks:
                        chunks_content.append(chunk.content)
                        if hasattr(chunk, "embedding") and chunk.embedding:
                            embeddings.append(chunk.embedding)
                
                texts.append(clean_content)
                documents.append({
                    "content": clean_content,  # This is the cleaned data
                    "source": source,
                    "structured_data": doc_structured,  # This is the structured extraction result
                    "embeddings": embeddings,  # Embeddings from embedding processor
                    "chunks": chunks_content,  # Chunk contents for reference
                    "metadata": {
                        "from_pipeline": True,
                        "title": getattr(doc.meta, "title", ""),
                        "url": source
                    },
                })
                
                # Debug output
                if doc_structured:
                    print(f"\n✓ Document {doc.id} has structured data: {doc_structured}")
                if embeddings:
                    print(f"✓ Document {doc.id} has {len(embeddings)} embeddings from processor")

            if documents:
                print(f"\nAdding {len(documents)} documents to RAG with cleaned data and structured extraction...\n")
                doc_ids = self.rag_manager.add_crawled_documents(documents)
                try:
                    self.rag_manager.build_and_save()
                except Exception as e:
                    logger.warning(f"Failed to build/save RAG index: {e}")
                
                # Save WebisDocuments as JSON
                self._save_webis_documents(result.documents, query)
                
                return doc_ids

        except Exception as e:
            logger.warning(f"Pipeline execution failed, falling back to simulated data: {e}")

        # Fallback: simulated data when pipeline cannot run
        texts = [
            "Example: NVIDIA releases new generation AI accelerator with significant performance improvement.",
            "Example: Multiple vendors launch customized AI inference chips, optimizing large model inference.",
        ]
        clean_rows = [{"text_file": f"sim_{i}.txt"} for i in range(len(texts))]

        documents = []
        for i, txt in enumerate(texts):
            source = f"pipeline_{i}"
            if i < len(clean_rows):
                tf = clean_rows[i].get("text_file")
                if tf:
                    try:
                        source = Path(tf).stem
                    except Exception:
                        pass

            documents.append(
                {
                    "content": txt,
                    "source": source,
                    "structured_data": None,
                    "metadata": {"from_pipeline": True},
                }
            )

        if documents:
            print("Adding fallback documents to RAG...\n")
            doc_ids = self.rag_manager.add_crawled_documents(documents)
            try:
                self.rag_manager.build_and_save()
            except Exception:
                pass
            
            # Save fallback documents as JSON
            try:
                from datetime import datetime
                output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "webis_documents"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                query_safe = "".join(c if c.isalnum() else "_" for c in query)[:50]
                output_file = output_dir / f"webis_docs_fallback_{query_safe}_{timestamp}.json"
                
                fallback_data = {
                    "query": query,
                    "timestamp": timestamp,
                    "document_count": len(documents),
                    "type": "fallback",
                    "documents": documents,
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(fallback_data, f, ensure_ascii=False, indent=2)
                
                print(f"✓ Saved fallback documents to {output_file}\n")
                logger.info(f"Saved fallback documents to {output_file}")
            except Exception as e:
                logger.warning(f"Failed to save fallback documents: {e}")
            
            return doc_ids

        return []


def example_webis_fetcher_stub(query: str) -> List[Dict[str, str]]:
    """Example webis_fetcher placeholder: In real use, replace with implementation that calls crawler/search/external pipeline.

    Return value format example:
    [
        {"content": "Cleaned text...", "source": "source_name"},
        ...
    ]
    """
    # 简单示例：返回空列表表示没有新数据
    logger.info(f"Stub webis_fetcher called for query: {query}")
    return []


def example_usage():
    """最小示例：展示如何创建Agent并处理query。"""
    try:
        from .llm import get_default_llm

        llm = get_default_llm()
    except Exception:
        llm = None

    agent = WebisRAGAgent(llm=llm)

    res = agent.handle_query("What are the recent advancements in AI chips?", force_refresh=False)
    print("Query result:", res)


if __name__ == "__main__":
    example_usage()


# Note: external fetcher factories have been removed. Agent uses internal pipeline invocation only.
