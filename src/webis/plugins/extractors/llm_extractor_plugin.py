"""
LLM Extractor Plugin for Webis v2.
"""

import json
import logging
import re
from typing import List, Optional, Any, Dict, Union

from webis.core.plugin import ExtractorPlugin
from webis.core.schema import WebisDocument, PipelineContext, StructuredResult, Lineage
from webis.core.llm.base import get_default_router

logger = logging.getLogger(__name__)


class LLMExtractorPlugin(ExtractorPlugin):
    """
    Extracts structured data from documents using an LLM.
    """
    
    name = "llm_extractor"
    description = "Extract structured data using Large Language Models"
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.output_format = self.config.get("output_format", "json")
        self.schema = self.config.get("schema", None)

    def extract(
        self,
        docs: List[WebisDocument],
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> "StructuredResult":
        
        task = context.task if context else "Extract information"
        
        # Combine text
        combined_text = ""
        doc_ids = []
        for doc in docs:
            content = doc.clean_content or doc.content or ""
            # Limit content length to avoid context window overflow (naive truncation)
            # In a real system, we would chunk and map-reduce.
            if len(content) > 50000:
                content = content[:50000] + "...(truncated)"
            
            combined_text += f"--- Document Source: {doc.meta.url or doc.id} ---\n{content}\n\n"
            doc_ids.append(doc.id)
            
        if not combined_text:
             logger.warning("No content to extract from.")
             return StructuredResult(
                 schema_id="empty",
                 data={},
                 lineage=Lineage(source_doc_ids=[])
             )

        # Build Prompt
        prompt = f"""
        You are an expert data analyst. 
        Your task is: {task}
        
        Extract relevant information from the provided text into a structured JSON format.
        """
        
        if self.schema:
            prompt += f"\nEvaluate against this schema:\n{json.dumps(self.schema, indent=2)}\n"
        else:
            prompt += "\n Infer the best structure for this data. Return a key-value JSON object.\n"
            
        prompt += f"\nInput Text:\n{combined_text}\n"
        prompt += "\nReturn ONLY the JSON object."
        
        router = get_default_router()
        
        try:
            response = router.chat(
                [{"role": "user", "content": prompt}],
                model=None, # Use primary model (e.g. DeepSeek-V3.2)
                temperature=0.0,
                supports_json_mode=True
            )
            
            raw = response.content
            
            # Parse JSON
            data = {}
            match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {"raw_text": raw, "error": "JSON parse failed"}
            else:
                data = {"raw_text": raw}
                
            return StructuredResult(
                schema_id="dynamic_llm_extraction",
                data=data,
                raw_output=raw,
                lineage=Lineage(
                    source_doc_ids=doc_ids,
                    model_name=response.model
                )
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return StructuredResult(
                 schema_id="error",
                 data={"error": str(e)},
                 lineage=Lineage(source_doc_ids=doc_ids),
                 is_valid=False
            )
