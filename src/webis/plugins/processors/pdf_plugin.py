"""
PDF Processor Plugin for Webis v2.
"""

import logging
import os
import tempfile
import requests
from typing import Optional, List, Any

from webis.core.plugin import ProcessorPlugin
from webis.core.schema import WebisDocument, DocumentType, PipelineContext
from webis.core.llm.base import get_default_router

logger = logging.getLogger(__name__)


class PDFPlugin(ProcessorPlugin):
    """
    Extracts text from PDF files using PyPDFLoader and applies AI denoising.
    """
    
    name = "pdf_extractor"
    description = "Extract and clean text from PDF files"
    supported_types = ["pdf"]
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 3000)
    
    def process(
        self, 
        doc: WebisDocument, 
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> Optional[WebisDocument]:
        
        if doc.doc_type != DocumentType.PDF:
            return doc
            
        # 1. Prepare file
        temp_file = None
        file_path = ""
        
        try:
            # Check if content is a file path
            if doc.content and os.path.exists(doc.content):
                file_path = doc.content
            # Check meta
            elif doc.meta.custom.get("file_path") and os.path.exists(doc.meta.custom.get("file_path")):
                file_path = doc.meta.custom.get("file_path")
            # If URL exists, download it
            elif doc.meta.url:
                logger.info(f"Downloading PDF from {doc.meta.url}...")
                response = requests.get(doc.meta.url, timeout=30)
                response.raise_for_status()
                
                fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                os.write(fd, response.content)
                os.close(fd)
                temp_file = temp_path
                file_path = temp_path
            else:
                logger.warning(f"No content or URL for PDF document {doc.id}")
                return doc

            # 2. Extract text
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except ImportError:
                logger.error("Missing dependency: pip install langchain-community pypdf")
                doc.add_processing_step(self.name, {"status": "failed", "error": "Missing dependencies"})
                return doc

            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_parts = []
            for i, page in enumerate(pages, 1):
                page_text = page.page_content.strip()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")
            
            raw_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(raw_text)} chars from PDF")
            
            # 3. AI Denoise
            cleaned_text = self._ai_denoise(raw_text)
            
            doc.clean_content = cleaned_text
            doc.add_processing_step(self.name, {"status": "success", "pages": len(pages)})
            
            return doc
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            doc.add_processing_step(self.name, {"status": "failed", "error": str(e)})
            return doc
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _ai_denoise(self, text: str) -> str:
        """Apply AI denoising to text."""
        # Split text into chunks
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        denoised_chunks = []
        
        router = get_default_router()
        
        for chunk in chunks:
            prompt = f"""请对以下PDF提取文本进行通用降噪优化，严格遵循以下要求：
1. 去除冗余信息：重复页眉页脚、无效特殊符号、连续空白行/空格；
2. 统一格式规范：全角字符转半角、标点符号统一、文本断行修复；
3. 保留核心内容：分页标记（--- Page X ---）、所有有意义文本；
4. 修复文本问题：文字错乱、拼写错误，不修改核心语义；
5. 仅输出优化后文本，不添加额外解释。

文本内容：{chunk}"""

            try:
                response = router.chat(
                    [{"role": "user", "content": prompt}],
                    model=None, # Prefer fast/cheap model or high quality
                    temperature=0.1
                )
                denoised_chunks.append(response.content)
            except Exception as e:
                logger.warning(f"Denoise failed for chunk: {e}")
                denoised_chunks.append(chunk) # Fallback to original
                
        return "\n\n".join(denoised_chunks)
