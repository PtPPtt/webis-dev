from webis.core.plugin import get_default_registry
from .html_report_plugin import HtmlReportPlugin

__all__ = ["HtmlReportPlugin"]

# Auto-register
get_default_registry().register(HtmlReportPlugin())
