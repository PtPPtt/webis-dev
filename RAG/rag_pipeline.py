"""
RAG Pipeline Module - Core RAG Processing Pipeline

Handles:
- Document fetching from webis pipeline
- Document chunking
- Embedding generation
- Document storage and retrieval
- Returns structured retrieval results for downstream tasks
"""

from typing import Any, Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np

from chunker import ChunkingPipeline, Chunk
from embedding_processor import EmbeddingGemmaPlugin
from rag_tools import RAGManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Independent RAG Pipeline Module
    
    Responsibilities:
    - Fetch documents using webis pipeline
    - Chunk documents into manageable pieces
    - Generate embeddings for chunks
    - Store documents in vector database
    - Retrieve relevant documents for queries
    
    Output: Structured retrieval results that can be used by downstream tasks
    """
    
    def __init__(
        self,
        rag_store_path: str = "./data/rag_store.json",
        chunk_strategy: str = "sliding_window",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model_type: str = "gemma",
        top_k: int = 3,
        min_doc_threshold: int = 1,
        min_score_threshold: float = 0.4,
    ):
        """
        Initialize RAG Pipeline.
        
        Args:
            rag_store_path: Path to RAG storage
            chunk_strategy: Chunking strategy
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks
            embedding_model_type: Type of embedding model
            top_k: Default number of documents to retrieve
            min_doc_threshold: Minimum number of documents required before fetching from webis
            min_score_threshold: Minimum relevance score threshold for documents
        """
        self.rag_store_path = rag_store_path
        self.top_k = top_k
        self.min_doc_threshold = min_doc_threshold
        self.min_score_threshold = min_score_threshold
        
        # Initialize embedding processor
        try:
            self.embedding_processor = EmbeddingGemmaPlugin(
                model_type=embedding_model_type,
                device="cpu"
            )
            logger.info(f"✓ Initialized {embedding_model_type} embedding processor")
        except Exception as e:
            self.embedding_processor = None
            logger.warning(f"Failed to initialize embedding processor: {e}")
        
        # Initialize chunking pipeline
        self.chunking_pipeline = ChunkingPipeline(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            separator="\n\n",
        )
        
        # Initialize RAG manager with embedding processor
        self.rag_manager = RAGManager(
            rag_store_path=rag_store_path,
            embedding_processor=self.embedding_processor
        )
    
    def process_and_store_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str = None,
    ) -> Dict[str, Any]:
        """
        Process documents through pipeline: chunk -> embed -> store.
        
        Args:
            documents: List of documents from webis pipeline, each containing:
                - content: str
                - source: str
                - structured_data: dict (optional)
                - metadata: dict (optional)
            query: Optional query for context (for logging)
            
        Returns:
            {
                "processed_count": int,
                "chunk_count": int,
                "doc_ids": [str],
                "embedding_count": int,
                "documents": [{processed doc info}]
            }
        """
        if not documents:
            logger.warning("No documents to process")
            return {
                "processed_count": 0,
                "chunk_count": 0,
                "doc_ids": [],
                "embedding_count": 0,
                "documents": []
            }
        
        processed_docs = []
        total_chunks = 0
        total_embeddings = 0
        
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            
            if not content or not content.strip():
                logger.debug(f"Skipping empty document from {source}")
                continue
            
            # ===== CHUNKING =====
            chunk_metadata = {
                "source": source,
                "title": doc.get("title", ""),
            }
            chunks = self.chunking_pipeline.process_single(content, metadata=chunk_metadata)
            
            if not chunks:
                logger.debug(f"Document {source} produced no chunks")
                continue
            
            total_chunks += len(chunks)
            logger.info(f"✓ Document {source} chunked into {len(chunks)} chunks")
            
            # ===== EMBEDDING =====
            embeddings = []
            if self.embedding_processor:
                try:
                    chunk_texts = [chunk.content for chunk in chunks]
                    embedding_vectors = self.embedding_processor.embed_texts(chunk_texts)
                    
                    for i, chunk in enumerate(chunks):
                        if embedding_vectors[i] is not None:
                            chunk.embedding = embedding_vectors[i]
                            embeddings.append(embedding_vectors[i])
                    
                    total_embeddings += len(embeddings)
                    logger.info(f"✓ Generated {len(embeddings)} embeddings for {source}")
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")
            
            processed_docs.append({
                "content": content,
                "source": source,
                "structured_data": doc.get("structured_data"),
                "embeddings": embeddings,
                "chunks": chunks,
                "metadata": doc.get("metadata", {}),
            })
        
        # ===== STORE =====
        doc_ids = []
        if processed_docs:
            doc_ids = self.rag_manager.add_crawled_documents(processed_docs)
            try:
                self.rag_manager.build_and_save()
                logger.info(f"✓ Stored {len(doc_ids)} documents to RAG")
            except Exception as e:
                logger.warning(f"Failed to build/save RAG: {e}")
        
        return {
            "processed_count": len(processed_docs),
            "chunk_count": total_chunks,
            "doc_ids": doc_ids,
            "embedding_count": total_embeddings,
            "documents": processed_docs,
        }
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents relevant to query.
        
        Note: This method only retrieves from existing RAG, without auto-fetching.
        Use get_retrieval_context() with auto_fetch_webis=True for auto-fetching functionality.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            {
                "query": str,
                "top_k": int,
                "documents": [{doc info}],
                "context": str,
                "scores": [float],
                "total_retrieved": int
            }
        """
        if top_k is None:
            top_k = self.top_k
        
        result = self.rag_manager.retrieve_for_query(
            query=query,
            top_k=top_k,
            include_scores=True
        )
        
        return {
            "query": query,
            "top_k": top_k,
            "documents": result.get("documents", []),
            "context": result.get("context", ""),
            "scores": result.get("scores", []),
            "total_retrieved": len(result.get("documents", [])),
        }
    
    def _fetch_from_webis(self, query: str) -> bool:
        """
        Fetch documents from webis pipeline when retrieval results are insufficient.
        
        Args:
            query: Query text for webis search
            
        Returns:
            True if documents were fetched and stored, False otherwise
        """
        try:
            print(f"\n⚠️  Insufficient documents in RAG )")
            print(f"📡 Fetching from webis pipeline for query: '{query}'...\n")
            
            from webis.core.pipeline import Pipeline
        except ImportError:
            logger.warning("Webis pipeline not available, skipping fetch")
            return False
        
        try:
            # Initialize webis pipeline
            pipe = Pipeline()
            
            # Register plugins
            try:
                from webis.plugins.sources import DuckDuckGoPlugin
                from webis.plugins.processors import (
                    HtmlFetcherPlugin,
                    HtmlCleanerPlugin,
                )
                from webis.plugins.extractors import LLMExtractorPlugin
                
                registry = pipe.registry
                try:
                    registry.register_class(DuckDuckGoPlugin)
                except:
                    pass
                try:
                    registry.register_class(HtmlFetcherPlugin)
                except:
                    pass
                try:
                    registry.register_class(HtmlCleanerPlugin)
                except:
                    pass
                try:
                    registry.register_class(LLMExtractorPlugin)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Some plugins unavailable: {e}")
            
            # Build pipeline
            pipe.add_source("duckduckgo", max_results=5)
            pipe.add_processor("html_fetcher")
            pipe.add_processor("html_cleaner")
            pipe.add_extractor("llm_extractor", config={
                "output_format": "json",
                "schema": {
                    "title": "string",
                    "key_points": ["string"],
                    "summary": "string"
                }
            })
            
            # Run pipeline
            result = pipe.run(query)
            
            if not result or not result.documents:
                logger.warning("Webis returned no documents")
                return False
            
            print(f"✓ Webis fetched {result.document_count} documents\n")
            
            # Process webis results
            documents = []
            for doc in result.documents:
                clean_content = getattr(doc, "clean_content", None) or getattr(doc, "content", "")
                if clean_content and clean_content.strip():
                    documents.append({
                        "content": clean_content,
                        "source": getattr(doc.meta, "url", None) or getattr(doc.meta, "title", "unknown"),
                        "title": getattr(doc.meta, "title", ""),
                        "structured_data": None,
                        "metadata": {
                            "from_webis": True,
                            "timestamp": datetime.now().isoformat(),
                        }
                    })
            
            if documents:
                # Process and store documents
                print(f"Processing and storing {len(documents)} documents to RAG...\n")
                result = self.process_and_store_documents(documents, query=query)
                
                print(f"✓ Stored {result['processed_count']} documents")
                print(f"✓ Generated {result['chunk_count']} chunks")
                print(f"✓ Generated {result['embedding_count']} embeddings\n")
                
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to fetch from webis: {e}")
            return False
    
    def _should_fetch_webis(self, retrieval_result: Dict[str, Any]) -> bool:
        """
        Determine if we should fetch from webis based on retrieval results.
        
        Args:
            retrieval_result: Result from retrieve() method
            
        Returns:
            True if should fetch, False otherwise
        """
        doc_count = len(retrieval_result.get("documents", []))
        scores = retrieval_result.get("scores", [])
        
        # Check if document count is below threshold
        if doc_count < self.min_doc_threshold:
            return True
        
        # Check if top score is below threshold
        if scores and min(scores) < self.min_score_threshold:
            return True
        
        return False
    
    def get_retrieval_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        auto_fetch_webis: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for downstream tasks.
        
        Automatically fetches from webis if retrieval results are insufficient.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            auto_fetch_webis: Automatically fetch from webis if results insufficient
            
        Returns:
            {
                "query": str,
                "retrieved_documents": [...],
                "context_text": str,
                "structured_data": {...},
                "metadata": {...},
                "webis_fetched": bool
            }
        """
        retrieval_result = self.retrieve(query, top_k)
        
        # Check if we need to fetch from webis
        webis_fetched = False
        if auto_fetch_webis and self._should_fetch_webis(retrieval_result):
            webis_fetched = self._fetch_from_webis(query)
            
            # Re-retrieve after fetching
            if webis_fetched:
                retrieval_result = self.retrieve(query, top_k)
        
        # Get full RAG context for better formatting
        context_data = self.rag_manager.rag.retrieve_context(
            query=query,
            top_k=top_k or self.top_k
        )
        
        return {
            "query": query,
            "retrieved_documents": retrieval_result["documents"],
            "context_text": context_data["context"],
            "structured_data": context_data["structured_data"],
            "scores": retrieval_result["scores"],
            "metadata": {
                "retrieval_count": len(retrieval_result["documents"]),
                "top_k": retrieval_result["top_k"],
                "webis_fetched": webis_fetched,
            }
        }
    
    def save(self):
        """Save RAG state to disk"""
        try:
            self.rag_manager.build_and_save()
            logger.info(f"✓ RAG Pipeline saved to {self.rag_store_path}")
        except Exception as e:
            logger.warning(f"Failed to save RAG: {e}")
    
    def load(self):
        """Load RAG state from disk"""
        try:
            self.rag_manager._try_load()
            logger.info(f"✓ RAG Pipeline loaded from {self.rag_store_path}")
        except Exception as e:
            logger.warning(f"Failed to load RAG: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        return self.rag_manager.get_stats()
    
    def display_stats(self):
        """Display RAG pipeline statistics"""
        self.rag_manager.display_stats()
