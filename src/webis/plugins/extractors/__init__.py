from .llm_extractor_plugin import LLMExtractorPlugin

__all__ = [
    "LLMExtractorPlugin"
]

# Auto-register plugins
from webis.core.plugin import get_default_registry

registry = get_default_registry()
registry.register(LLMExtractorPlugin())
