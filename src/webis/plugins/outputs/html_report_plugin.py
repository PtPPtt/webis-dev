
import json
import os
import re
import datetime
from typing import List, Union, Any, Dict, Optional

from webis.core.plugin import OutputPlugin
from webis.core.schema import WebisDocument, StructuredResult, PipelineContext
from webis.core.llm.base import get_default_router

class HtmlReportPlugin(OutputPlugin):
    """
    Generates a beautiful HTML report from pipeline results using LLM generation.
    """
    name = "html_report"
    description = "Generates a standalone HTML report by asking an LLM to render the data."

    def save(
        self,
        data: Union[StructuredResult, List[Any]],
        context: Optional[PipelineContext] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> bool:
        if not output_dir:
            return False
            
        # Prepare content for the LLM
        result_data = {}
        documents = []
        task_name = "Webis Task"
        
        if context:
            task_name = context.task
            
        if isinstance(data, StructuredResult):
            result_data = data.data
            documents = kwargs.get("documents", [])
        elif isinstance(data, list):
            if data and isinstance(data[0], WebisDocument):
                documents = data
                result_data = {"info": "Raw document list"}
            else:
                result_data = {"data": data}
        else:
             result_data = data
             
        # Generate HTML using LLM
        html_content = self._generate_with_llm(task_name, result_data, documents)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report.html")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return True

    def _generate_with_llm(self, task: str, data: Dict, docs: List[WebisDocument]) -> str:
        """
        Uses LLM to write the HTML code dynamically.
        """
        # 1. Prepare Content Context
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Summarize sources to avoid hitting token limits
        sources_summary = []
        for i, doc in enumerate(docs[:10]): # Limit to top 10 sources
            title = doc.meta.title or f"Source {i+1}"
            url = doc.meta.url or "#"
            sources_summary.append(f"- [{title}]({url})")
        
        sources_text = "\n".join(sources_summary)

        # 2. Construct Prompt (Adapted from legacy_v1/visual.py)
        system_prompt = """你是一个专业的网页开发者。请生成完整、格式正确、可渲染的HTML5代码。

重要要求：
1) 必须生成完整HTML文档：以<!DOCTYPE html>开头，以</html>结束
2) 所有标签必须正确闭合；不要输出 <p >、</ div > 这类“尖括号附近有空格”的标签
3) 标签名/属性名必须为英文半角；属性值必须使用英文双引号 " 包裹（不要用中文引号“”）
4) CSS/JS 必须使用英文半角标点（: ; ( ) , .），不要出现中文全角标点（例如：：；（））
5) 不要输出类似 <<div>、<<footer> 这种重复尖括号的标签
6) 不要输出 ```html 这类Markdown代码块包裹
7) 只输出HTML代码，不要额外说明；输出前请自检并修复以上问题"""

        user_content = f"""
任务目标: {task}

核心数据 (JSON):
{json_str}

参考来源:
{sources_text}
"""

        user_prompt = f"""请为我创建美观的HTML网页，要求：

设计规范：
1. 完整HTML5结构
2. 现代化浅色UI配色 (Modern UI)
3. 合理的内容分区，网页结构不要过于简单
4. 优雅的排版，使用卡片式布局展示“核心数据”
5. 必须在显眼位置展示“任务目标”
6. 在底部列出“参考来源”链接
7. 内容来源只能使用“基于Webis生成”标志
8. 可以添加合理配图或图表（使用简单的CSS/SVG绘制）

内容来源（合理总结使用）：
{user_content}

重要：只输出HTML代码，不要额外说明。"""

        # 3. Call LLM
        try:
            router = get_default_router()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Using the primary model via the router
            response = router.chat(messages)
            raw_html = response.content
            
        except Exception as e:
            # Fallback if generation fails
            print(f"LLM Generation failed: {e}")
            return f"<html><body><h1>Error Generating Report</h1><p>{e}</p></body></html>"

        # 4. Sanitize and Return
        return self._sanitize_html(raw_html)

    def _strip_markdown_fences(self, text: str) -> str:
        stripped = text.strip()
        m = re.search(r"```(?:html)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()[1:]
            stripped = "\n".join(lines)
            stripped = re.sub(r"\n```\s*$", "", stripped)
            return stripped.strip()
        return stripped

    def _sanitize_html(self, html: str) -> str:
        s = self._strip_markdown_fences(html)

        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # Fix common model glitch patterns like '<< section ...' and '< / div >'
        s = re.sub(r"<<\s*/\s*([A-Za-z])", r"</\1", s)
        s = re.sub(r"<<\s*([A-Za-z])", r"<\1", s)

        s = re.sub(r"<\s*/\s*([A-Za-z][\w:-]*)\s*>", r"</\1>", s)
        s = re.sub(r"<\s+([!/A-Za-z])", r"<\1", s)

        attrs = r"(?:class|id|href|src|style|lang|type|rel|name|content|crossorigin|role)"
        s = re.sub(rf"(<\s*/?\s*)([A-Za-z][\w:-]*?)({attrs})\s*=", r"\1\2 \3=", s)
        s = re.sub(r"(<\s*/?\s*)([A-Za-z][\w:-]*?)(aria-[\w-]+)\s*=", r"\1\2 \3=", s)

        s = re.sub(r'(\s[\w:-]+)\s*=\s*"', r'\1="', s)

        def _normalize_style_block(m: re.Match) -> str:
            style = m.group(1)
            style = (
                style.replace("：", ":")
                .replace("；", ";")
                .replace("（", "(")
                .replace("）", ")")
                .replace("，", ",")
            )
            style = re.sub(r"\b(var|calc|rgba|rgb)\s*\(\s*", r"\1(", style)
            style = re.sub(r"\(\s*--", "(--", style)
            return f"<style>{style}</style>"

        s = re.sub(r"<style[^>]*>(.*?)</style>", _normalize_style_block, s, flags=re.DOTALL | re.IGNORECASE)
        return s.strip()
