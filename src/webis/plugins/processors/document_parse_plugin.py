"""
Document Parse Plugin for Webis v2.
"""

import logging
import os
import tempfile
import requests
import pathlib
from typing import Optional, List, Any

from webis.core.plugin import ProcessorPlugin
from webis.core.schema import WebisDocument, DocumentType, PipelineContext
from webis.core.llm.base import get_default_router

logger = logging.getLogger(__name__)


class DocumentParsePlugin(ProcessorPlugin):
    """
    Extracts text from DOCX, TXT, and MD files using LangChain loaders and applies AI denoising.
    """
    
    name = "document_parser"
    description = "Extract and clean text from DOCX, TXT, and MD files"
    supported_types = ["docx", "doc", "txt", "md", "markdown"]
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 3000)
    
    def process(
        self, 
        doc: WebisDocument, 
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> Optional[WebisDocument]:
        
        # Check support based on doc_type or extension if available
        # Ideally doc.doc_type should be set. If UNKNOWN, check if content is a path with extension.
        
        should_process = False
        if doc.doc_type.value in self.supported_types:
            should_process = True
        
        file_path = ""
        # 1. Prepare file
        temp_file = None
        
        if doc.content and os.path.exists(doc.content):
            file_path = doc.content
        elif doc.meta.custom.get("file_path") and os.path.exists(doc.meta.custom.get("file_path")):
            file_path = doc.meta.custom.get("file_path")
        elif doc.meta.url:
             # Need to download to know extension if not clear?
             # Or assume user set doc_type correctly.
             pass
        
        if file_path:
             ext = pathlib.Path(file_path).suffix.lower().lstrip(".")
             if ext in self.supported_types:
                 should_process = True
        
        if not should_process and not (doc.meta.url and any(doc.meta.url.lower().endswith(f".{t}") for t in self.supported_types)):
            return doc
            
        try:
            # Download if needed
            if not file_path and doc.meta.url:
                logger.info(f"Downloading document from {doc.meta.url}...")
                response = requests.get(doc.meta.url, timeout=30)
                response.raise_for_status()
                
                # Try to guess extension from URL or Content-Type
                ext = ".txt" # default
                if "." in doc.meta.url.split("/")[-1]:
                     ext = "." + doc.meta.url.split("/")[-1].split(".")[-1]
                
                fd, temp_path = tempfile.mkstemp(suffix=ext)
                os.write(fd, response.content)
                os.close(fd)
                temp_file = temp_path
                file_path = temp_path
            
            if not file_path:
                 return doc

            # 2. Extract text
            try:
                from langchain_community.document_loaders import Docx2txtLoader, TextLoader
            except ImportError:
                logger.error("Missing dependency: pip install langchain-community docx2txt")
                doc.add_processing_step(self.name, {"status": "failed", "error": "Missing dependencies"})
                return doc

            ext = pathlib.Path(file_path).suffix.lower()
            
            if ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                
            docs = loader.load()
            raw_text = "\n".join(d.page_content for d in docs).strip()
            
            logger.info(f"Extracted {len(raw_text)} chars from {ext} file")
            
            # 3. AI Denoise
            cleaned_text = self._ai_denoise(raw_text, is_markdown=(ext in [".md", ".markdown"]))
            
            doc.clean_content = cleaned_text
            doc.add_processing_step(self.name, {"status": "success", "length": len(cleaned_text)})
            
            # Update doc_type if needed
            if ext in [".docx", ".doc"]:
                 doc.doc_type = DocumentType.UNKNOWN # No DOCX type in schema yet? mapped to TEXT or UNKNOWN
            elif ext in [".md", ".markdown"]:
                 doc.doc_type = DocumentType.MARKDOWN
            elif ext == ".txt":
                 doc.doc_type = DocumentType.TEXT
            
            return doc
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            doc.add_processing_step(self.name, {"status": "failed", "error": str(e)})
            return doc
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _ai_denoise(self, text: str, is_markdown: bool = False) -> str:
        """Apply AI denoising to text."""
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        denoised_chunks = []
        
        router = get_default_router()
        
        for chunk in chunks:
            prompt = f"""请对以下文档提取文本进行通用降噪优化，严格遵循以下要求：
1. 表格专项处理：
   - docx表格：修复内容错乱，保持行列结构清晰，用竖线|分隔单元格；
   - Markdown表格：{'移除MD表格语法，转为纯文本行列结构' if is_markdown else '自动识别并规范化'};
2. Markdown语法清洗：
   - {'移除所有MD语法符号（标题#、加粗**、链接[]()等），仅保留纯文本' if is_markdown else '如果存在Markdown语法，请清理只保留文本'};
3. 通用降噪规则：
   - 去除冗余信息：重复页眉页脚、无效特殊符号；
   - 统一格式规范：全角字符转半角、标点符号统一；
4. 仅输出优化后纯文本，不添加额外解释。

文本内容：{chunk}"""

            try:
                response = router.chat(
                    [{"role": "user", "content": prompt}],
                    model=None,
                    temperature=0.1
                )
                denoised_chunks.append(response.content)
            except Exception as e:
                logger.warning(f"Denoise failed for chunk: {e}")
                denoised_chunks.append(chunk)
                
        return "\n\n".join(denoised_chunks)
