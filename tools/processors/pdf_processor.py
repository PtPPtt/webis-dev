#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Processor
Process PDF files
"""
import logging
import os
import requests
from typing import Dict, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base_processor import BaseFileProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseFileProcessor):
    """PDF Processor - Process PDF files"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.pdf'}

    def get_processor_name(self) -> str:
        return "PDFProcessor"

    def get_supported_extensions(self) -> set:
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
        """AI降噪：优化文本处理逻辑，添加重试机制"""
        try:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("[PDFProcessor] 未在环境变量中找到DEEPSEEK_API_KEY")
                return text

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # 修复文本块分割逻辑（步长与块大小保持一致，避免重叠）
            chunk_size = 3000
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            denoised_chunks = []

            for chunk in text_chunks:
                prompt = f"""请对以下PDF提取文本进行通用降噪优化，严格遵循以下要求：
1. 去除冗余信息：重复页眉页脚、无效特殊符号（占位符、乱码）、连续空白行/空格、无意义图表占位文本；
2. 统一格式规范：全角字符转半角、标点符号统一、文本断行修复（如"新一 轮"→"新一轮"）、页码/编号格式对齐；
3. 保留核心内容：分页标记（--- Page X ---）、所有有意义文本（正文、标题、列表、注释等）、原始逻辑结构；
4. 修复文本问题：文字错乱、排版错位、拼写错误，不修改原文核心语义和关键信息；
5. 适配各类PDF：无论文档类型，保持处理一致性，仅输出优化后文本，不添加额外解释。

文本内容：{chunk}"""

                api_url = "https://api.deepseek.com/v1/chat/completions"
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2048  # 补充最大生成token限制
                }

                # 延长超时时间至60秒，使用带重试的请求方法
                response = self._send_ai_request(api_url, payload, headers, timeout=60)
                response.raise_for_status()
                chunk_result = response.json()["choices"][0]["message"]["content"]
                denoised_chunks.append(chunk_result)

            return "\n\n".join(denoised_chunks)
        except Exception as e:
            logger.error(f"[PDFProcessor] AI降噪失败：{str(e)}")
            return text

    def extract_text(self, file_path: str) -> Dict[str, Union[str, bool]]:
        """Extract PDF text with optimized processing"""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            text_parts = []
            for i, doc in enumerate(docs, 1):
                page_text = doc.page_content.strip()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")

            raw_text = "\n\n".join(text_parts)
            logger.info(
                f"[PDFProcessor] 原始文本提取完成：{file_path}, pages: {len(docs)}, text length: {len(raw_text)}")

            text = self._ai_denoise(raw_text)
            if text != raw_text:
                logger.info(f"[PDFProcessor] AI降噪完成，降噪后文本长度：{len(text)}")
            else:
                logger.warning("[PDFProcessor] 未启用AI降噪（未配置环境变量DEEPSEEK_API_KEY或降噪失败）")

            logger.info(
                f"[PDFProcessor] Successfully processed PDF: {file_path}, pages: {len(docs)}, text length: {len(text)}")
            return {"success": True, "text": text, "error": ""}

        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}. Please install: pip install langchain-community pypdf requests tenacity"
            logger.error(f"[PDFProcessor] {error_msg}")
            return {"success": False, "text": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Failed to process PDF: {str(e)}"
            logger.error(f"[PDFProcessor] Failed to process PDF {file_path}: {str(e)}")
            return {"success": False, "text": "", "error": error_msg}
