# zyx ==============================================================================

from zyx.ext._loader import UtilLazyLoader


class MathPlugin(UtilLazyLoader):
    pass


MathPlugin.init("semantic_kernel.core_plugins.math_plugin", "MathPlugin")


class TimePlugin(UtilLazyLoader):
    pass


TimePlugin.init("semantic_kernel.core_plugins.time_plugin", "TimePlugin")


class TextPlugin(UtilLazyLoader):
    pass


TextPlugin.init("semantic_kernel.core_plugins.text_plugin", "TextPlugin")


class TextMemoryPlugin(UtilLazyLoader):
    pass


TextMemoryPlugin.init(
    "semantic_kernel.core_plugins.text_memory_plugin", "TextMemoryPlugin"
)


class WebSearchEnginePlugin(UtilLazyLoader):
    pass


WebSearchEnginePlugin.init(
    "semantic_kernel.core_plugins.web_search_engine_plugin", "WebSearchEnginePlugin"
)


class HttpPlugin(UtilLazyLoader):
    pass


HttpPlugin.init("semantic_kernel.core_plugins.http_plugin", "HttpPlugin")


class WaitPlugin(UtilLazyLoader):
    pass


WaitPlugin.init("semantic_kernel.core_plugins.wait_plugin", "WaitPlugin")


class SessionsPythonTool(UtilLazyLoader):
    pass


SessionsPythonTool.init(
    "semantic_kernel.core_plugins.sessions_python_tool", "SessionsPythonTool"
)
