#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image OCR Processor
Process image files and perform text recognition with noise reduction
"""

import logging
import re
from typing import Dict, Union, Optional
from .base_processor import BaseFileProcessor

logger = logging.getLogger(__name__)


class ImageProcessor(BaseFileProcessor):
    """Image OCR Processor - Uses EasyOCR with noise reduction"""
    
    def __init__(self, deepseek_api_key: Optional[str] = None):
        super().__init__()
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        self._reader = None
        self.deepseek_api_key = deepseek_api_key  # Optional API key for DeepSeek enhancement
    
    def get_processor_name(self) -> str:
        """Get processor name"""
        return "ImageProcessor"
    
    def get_supported_extensions(self) -> set:
        """Get supported file extensions"""
        return self.supported_extensions
    
    def _get_reader(self):
        """Lazy load EasyOCR to avoid loading model at startup"""
        if self._reader is None:
            try:
                import easyocr
                logger.info("[ImageProcessor] Initializing EasyOCR model...")
                self._reader = easyocr.Reader(['ch_sim', 'en'])
                logger.info("[ImageProcessor] EasyOCR model initialization completed")
            except ImportError:
                raise ImportError("Please install easyocr: pip install easyocr")
        return self._reader
    
    def _basic_noise_reduction(self, text: str) -> str:
        """
        Apply basic rule-based noise reduction to the extracted text.
        
        - Remove special characters except common punctuation
        - Fix common OCR errors (e.g., 'l'/'1', 'O'/'0', 'S'/'5')
        - Remove extra whitespace and empty lines
        - Remove very short lines (less than 3 characters) that might be noise
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Common OCR error corrections
        corrections = {
            '1': 'l',  # 1 -> l
            '0': 'O',  # 0 -> O (context-dependent, but common in text)
            '5': 'S',  # 5 -> S
            '8': 'B',  # 8 -> B
            # Add more as needed based on common errors
        }
        
        # Apply corrections
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        # Remove special characters, keep letters, numbers, common punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"()-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into lines and filter short noisy lines
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) >= 3]
        
        return '\n'.join(cleaned_lines)
    
    def _deepseek_enhance(self, text: str) -> str:
        """
        Optional: Use DeepSeek API to enhance noise reduction and correct OCR errors.
        Requires API key and 'requests' library.
        
        Args:
            text: Text after basic cleaning
            
        Returns:
            Enhanced text or original if API fails
        """
        if not self.deepseek_api_key:
            logger.warning("[ImageProcessor] DeepSeek API key not provided, skipping enhancement")
            return text
        
        try:
            import requests
            import json
            
            url = "https://api.deepseek.com/v1/chat/completions"  # Adjust if needed
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an OCR text corrector. Fix errors in the following text without adding or removing content."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            enhanced_text = result['choices'][0]['message']['content'].strip()
            
            logger.info("[ImageProcessor] DeepSeek enhancement successful")
            return enhanced_text
            
        except ImportError:
            logger.error("[ImageProcessor] Missing 'requests' library for DeepSeek. Install: pip install requests")
            return text
        except Exception as e:
            logger.error(f"[ImageProcessor] DeepSeek enhancement failed: {str(e)}")
            return text
    
    def extract_text(self, file_path: str, use_deepseek: bool = False) -> Dict[str, Union[str, bool]]:
        """
        Extract text from image with noise reduction
        
        Args:
            file_path: Image file path
            use_deepseek: Whether to use DeepSeek enhancement (optional)
            
        Returns:
            Dict containing: success(bool), text(str), error(str)
        """
        try:
            reader = self._get_reader()
            results = reader.readtext(file_path)
            
            # Extract all recognized text
            text_lines = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    text_lines.append(text.strip())
            
            raw_text = "\n".join(text_lines)
            
            # Apply basic noise reduction
            cleaned_text = self._basic_noise_reduction(raw_text)
            
            # Optional DeepSeek enhancement
            if use_deepseek:
                cleaned_text = self._deepseek_enhance(cleaned_text)
            
            logger.info(f"[ImageProcessor] Successfully processed image: {file_path}, recognized {len(text_lines)} lines, final text length: {len(cleaned_text)}")
            return {"success": True, "text": cleaned_text, "error": ""}
            
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}. Please install: pip install easyocr"
            logger.error(f"[ImageProcessor] {error_msg}")
            return {"success": False, "text": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Failed to process image: {str(e)}"
            logger.error(f"[ImageProcessor] Failed to process image {file_path}: {str(e)}")
            return {"success": False, "text": "", "error": error_msg}
