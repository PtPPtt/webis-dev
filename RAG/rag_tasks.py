"""
RAG Tasks Module - Task Framework for Downstream Processing

Defines task interface and concrete task implementations that operate on RAG retrieval results.
Tasks are plugins that can be chained together to process retrieved documents.
"""

from typing import Any, Dict, List, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RAGTask(ABC):
    """
    Base class for RAG downstream tasks.
    
    A task receives retrieval context from RAG pipeline and produces outputs
    that can be consumed by the next task or the end user.
    """
    
    def __init__(self, task_name: str):
        """
        Initialize task.
        
        Args:
            task_name: Unique task identifier
        """
        self.task_name = task_name
    
    @abstractmethod
    def execute(self, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task on RAG retrieval context.
        
        Args:
            rag_context: Context dict from RAG pipeline containing:
                - query: str
                - retrieved_documents: [...]
                - context_text: str
                - structured_data: {...}
                - scores: [float]
                - metadata: {...}
                
        Returns:
            Task-specific output dict
        """
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get task name"""
        return self.task_name


class PromptEnhancementTask(RAGTask):
    """
    Task: Enhance prompt with RAG context for LLM/Agent.
    
    Takes retrieval results and formats them into an enhanced prompt
    that can be passed to downstream agents for structured extraction
    or answer generation.
    """
    
    def __init__(self, llm_agent=None):
        """
        Initialize prompt enhancement task.
        
        Args:
            llm_agent: Optional LLM/extraction agent for executing enhanced prompt
        """
        super().__init__("prompt_enhancement")
        self.llm_agent = llm_agent
    
    def execute(self, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prompt enhancement task.
        
        Args:
            rag_context: RAG retrieval context
            
        Returns:
            {
                "task_name": "prompt_enhancement",
                "success": bool,
                "enhanced_prompt": str,
                "reference_count": int,
                "agent_result": {...} (optional if agent provided)
            }
        """
        query = rag_context.get("query", "")
        context_text = rag_context.get("context_text", "")
        structured_data = rag_context.get("structured_data", {})
        
        # Format structured data
        structured_text = ""
        if structured_data:
            structured_parts = []
            for key, data in structured_data.items():
                if isinstance(data, dict):
                    formatted_data = "\n".join(f"  - {k}: {v}" for k, v in data.items())
                    structured_parts.append(f"**{key}**:\n{formatted_data}")
                else:
                    structured_parts.append(f"**{key}**: {data}")
            structured_text = "\n\n".join(structured_parts)
        
        # Build enhanced prompt
        if context_text or structured_text:
            enhanced_prompt = f"""## Reference Information (Prior Knowledge from Data Pipeline)

You have access to the following retrieved documents and structured insights from our knowledge base. Use this information to provide accurate, well-informed responses.

### Retrieved Documents:
{context_text or "No relevant documents found."}

### Structured Data Insights:
{structured_text or "No structured data available."}

---

## User Query

{query}

### Instructions for Response:
- Reference the provided documents and structured data when relevant.
- Provide detailed, evidence-based answers using the prior knowledge above.
- Maintain consistency with the retrieved information."""
        else:
            enhanced_prompt = query
        
        result = {
            "task_name": self.get_name(),
            "success": True,
            "enhanced_prompt": enhanced_prompt,
            "reference_count": len(rag_context.get("retrieved_documents", [])),
            "query": query,
        }
        
        # Execute agent if provided
        if self.llm_agent:
            try:
                logger.info("Executing LLM agent with enhanced prompt...")
                agent_result = self.llm_agent.extract(
                    prompt=enhanced_prompt,
                    text="",
                    output_format="markdown"
                )
                
                result["agent_result"] = {
                    "success": agent_result.success,
                    "output_format": agent_result.output_format,
                    "parsed": agent_result.parsed,
                    "raw": agent_result.raw,
                    "error": agent_result.error,
                }
            except Exception as e:
                logger.warning(f"Failed to execute agent: {e}")
                result["agent_result"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return result


class DocumentExtractionTask(RAGTask):
    """
    Task: Extract specific information from retrieved documents.
    
    Focuses on extracting structured information from the retrieved
    document context without LLM processing.
    """
    
    def __init__(self, extraction_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize document extraction task.
        
        Args:
            extraction_schema: Schema for extraction (optional)
        """
        super().__init__("document_extraction")
        self.extraction_schema = extraction_schema or {
            "documents": ["title", "content", "source"],
            "structured_data": "all"
        }
    
    def execute(self, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from documents.
        
        Args:
            rag_context: RAG retrieval context
            
        Returns:
            {
                "task_name": "document_extraction",
                "success": bool,
                "extracted_documents": [...],
                "extracted_data": {...},
                "query": str
            }
        """
        documents = rag_context.get("retrieved_documents", [])
        structured_data = rag_context.get("structured_data", {})
        
        extracted_docs = []
        for doc in documents:
            extracted_doc = {
                "source": doc.get("source"),
                "content": doc.get("content", "")[:500],  # Truncate long content
                "structured_data": doc.get("structured_data"),
            }
            extracted_docs.append(extracted_doc)
        
        return {
            "task_name": self.get_name(),
            "success": True,
            "extracted_documents": extracted_docs,
            "extracted_data": structured_data,
            "query": rag_context.get("query", ""),
            "document_count": len(extracted_docs),
        }


class SummaryTask(RAGTask):
    """
    Task: Generate summary of retrieved documents.
    
    Creates a concise summary of the retrieved context for quick overview.
    """
    
    def __init__(self, max_summary_length: int = 500):
        """
        Initialize summary task.
        
        Args:
            max_summary_length: Maximum length of summary text
        """
        super().__init__("summarization")
        self.max_summary_length = max_summary_length
    
    def execute(self, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of retrieved documents.
        
        Args:
            rag_context: RAG retrieval context
            
        Returns:
            {
                "task_name": "summarization",
                "success": bool,
                "summary": str,
                "source_count": int,
                "key_points": [str]
            }
        """
        documents = rag_context.get("retrieved_documents", [])
        context_text = rag_context.get("context_text", "")
        
        # Extract first part as summary
        if context_text:
            summary = context_text[:self.max_summary_length]
            if len(context_text) > self.max_summary_length:
                summary += "..."
        else:
            summary = "No documents retrieved."
        
        # Extract key points from structured data
        key_points = []
        structured_data = rag_context.get("structured_data", {})
        for key, data in structured_data.items():
            if isinstance(data, dict) and "summary" in data:
                key_points.append(data["summary"])
            elif isinstance(data, dict) and "key_points" in data:
                key_points.extend(data.get("key_points", []))
        
        return {
            "task_name": self.get_name(),
            "success": True,
            "summary": summary,
            "source_count": len(documents),
            "key_points": key_points[:5],  # Top 5 key points
        }


class TaskPipeline:
    """
    Task Pipeline - Orchestrates execution of multiple RAG tasks.
    
    Allows chaining tasks together where each task receives the output
    of the previous task (plus the original RAG context).
    """
    
    def __init__(self):
        """Initialize task pipeline"""
        self.tasks: List[RAGTask] = []
    
    def add_task(self, task: RAGTask) -> "TaskPipeline":
        """
        Add task to pipeline.
        
        Args:
            task: RAGTask instance
            
        Returns:
            Self for chaining
        """
        self.tasks.append(task)
        logger.info(f"Added task: {task.get_name()}")
        return self
    
    def execute(self, rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all tasks in pipeline.
        
        Args:
            rag_context: Initial RAG retrieval context
            
        Returns:
            {
                "query": str,
                "rag_context": {...},
                "task_results": [...],
                "success": bool
            }
        """
        if not self.tasks:
            logger.warning("No tasks in pipeline")
            return {
                "query": rag_context.get("query", ""),
                "rag_context": rag_context,
                "task_results": [],
                "success": False
            }
        
        task_results = []
        for task in self.tasks:
            try:
                logger.info(f"Executing task: {task.get_name()}")
                result = task.execute(rag_context)
                task_results.append(result)
            except Exception as e:
                logger.error(f"Task {task.get_name()} failed: {e}")
                task_results.append({
                    "task_name": task.get_name(),
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "query": rag_context.get("query", ""),
            "rag_context": rag_context,
            "task_results": task_results,
            "success": all(r.get("success", False) for r in task_results if "success" in r)
        }
