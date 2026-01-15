from .baidu_plugin import BaiduSearchPlugin
from .duckduckgo_plugin import DuckDuckGoPlugin
from .github_plugin import GitHubSearchPlugin
from .gnews_plugin import GNewsPlugin
from .hackernews_plugin import HackerNewsPlugin
from .semantic_scholar_plugin import SemanticScholarPlugin
from .serpapi_plugin import SerpApiPlugin

__all__ = [
    "BaiduSearchPlugin",
    "DuckDuckGoPlugin",
    "GitHubSearchPlugin",
    "GNewsPlugin",
    "HackerNewsPlugin",
    "SemanticScholarPlugin",
    "SerpApiPlugin",
]

# Auto-register plugins
from webis.core.plugin import get_default_registry

registry = get_default_registry()
registry.register(BaiduSearchPlugin())
registry.register(DuckDuckGoPlugin())
registry.register(GitHubSearchPlugin())
registry.register(GNewsPlugin())
registry.register(HackerNewsPlugin())
registry.register(SemanticScholarPlugin())
registry.register(SerpApiPlugin())
