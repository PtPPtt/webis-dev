#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Processor
Process doc/docx/txt/md files
"""

import logging
import os
import requests
from typing import Dict, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base_processor import BaseFileProcessor

logger = logging.getLogger(__name__)


class DocumentProcessor(BaseFileProcessor):
    """Document Processor - Process doc/docx/txt/md files"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.doc', '.docx', '.txt', '.md'}

    def get_processor_name(self) -> str:
        """Get processor name"""
        return "DocumentProcessor"

    def get_supported_extensions(self) -> set:
        """Get supported file extensions"""
        return self.supported_extensions

    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 重试间隔: 2s, 4s, 8s
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    )
    def _send_ai_request(self, api_url, payload, headers, timeout):
        """带重试机制的API请求发送方法"""
        return requests.post(api_url, json=payload, headers=headers, timeout=timeout)

    def _ai_denoise(self, text: str) -> str:
        """AI降噪：重点处理docx表格+Markdown语法清洗，添加重试机制"""
        try:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("[DocumentProcessor] 未在环境变量中找到DEEPSEEK_API_KEY")
                return text

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # 文本分块处理（适应长文档）
            chunk_size = 3000
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            denoised_chunks = []

            for chunk in text_chunks:
                prompt = f"""请对以下文档提取文本进行通用降噪优化，严格遵循以下要求：
1. 表格专项处理：
   - docx表格：修复内容错乱，保持行列结构清晰，用竖线|分隔单元格，表头与内容对应；
   - Markdown表格：移除MD表格语法（竖线、短横线分隔符），转为纯文本行列结构，保留表格内容；
2. Markdown语法清洗（核心要求）：
   - 移除所有MD语法符号：标题符号(#)、加粗(**/*)、斜体(*/_)、链接([文本](链接))、图片(![](链接))、列表符号(-/*/1.)、代码块(```/`)、分隔线(---)、脚注、锚点等；
   - 仅保留MD中的纯文本内容，不保留任何MD格式语法，确保文本简洁可读；
3. 通用降噪规则：
   - 去除冗余信息：重复页眉页脚、无效特殊符号、连续空白行/空格、无意义占位文本；
   - 统一格式规范：全角字符转半角、标点符号统一、文本断行修复（如"新一 轮"→"新一轮"）；
   - 保留核心内容：所有有意义文本（正文、标题文本、列表文本、表格内容等）、原始逻辑结构；
   - 修复文本问题：文字错乱、排版错位、拼写错误，不修改原文核心语义和关键信息；
4. 输出要求：仅返回优化后的纯文本，不添加任何额外解释、备注或格式。

文本内容：{chunk}"""

                api_url = "https://api.deepseek.com/v1/chat/completions"
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2048
                }

                # 延长超时时间至60秒，使用带重试的请求方法
                response = self._send_ai_request(api_url, payload, headers, timeout=60)
                response.raise_for_status()
                chunk_result = response.json()["choices"][0]["message"]["content"]
                denoised_chunks.append(chunk_result)

            return "\n\n".join(denoised_chunks)
        except Exception as e:
            logger.error(f"[DocumentProcessor] AI降噪失败：{str(e)}")
            return text

    def extract_text(self, file_path: str) -> Dict[str, Union[str, bool]]:
        """
        Extract document text

        Args:
            file_path: File path

        Returns:
            Dict containing: success(bool), text(str), error(str)
        """
        try:
            from langchain_community.document_loaders import Docx2txtLoader, TextLoader
            import pathlib

            ext = pathlib.Path(file_path).suffix.lower()

            if ext in [".docx", ".doc"]:
                # 使用docx2txt加载器处理Word文档（保留表格文本结构）
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif ext in [".txt", ".md"]:
                # 处理纯文本/MD文件，指定UTF-8编码避免乱码
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                return {"success": False, "text": "", "error": f"Unsupported file type: {ext}"}

            # 合并所有文档内容，去除首尾空白
            raw_text = "\n".join(doc.page_content for doc in docs).strip()
            logger.info(
                f"[DocumentProcessor] 原始文本提取完成：{file_path}, 文本长度：{len(raw_text)}")

            # 进行AI降噪处理（重点：MD语法清洗+表格修复）
            text = self._ai_denoise(raw_text)
            if text != raw_text:
                logger.info(f"[DocumentProcessor] AI降噪完成，降噪后文本长度：{len(text)}")
            else:
                logger.warning("[DocumentProcessor] 未启用AI降噪（未配置环境变量DEEPSEEK_API_KEY或降噪失败）")

            logger.info(
                f"[DocumentProcessor] Successfully processed document: {file_path}, text length: {len(text)}")
            return {"success": True, "text": text, "error": ""}

        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}. Please install: pip install langchain-community docx2txt requests tenacity"
            logger.error(f"[DocumentProcessor] {error_msg}")
            return {"success": False, "text": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Failed to process document: {str(e)}"
            logger.error(f"[DocumentProcessor] Failed to process document {file_path}: {str(e)}")
            return {"success": False, "text": "", "error": error_msg}
