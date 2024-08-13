# zyx ==============================================================================

_all__ = [
    "Toolkit",
    "Tool",
    "Calculator",
    "ApifyTools",
    "DuckDuckGo",
    "CsvTools",
    "ArxivToolkit",
    "WikipediaTools",
    "WebsiteTools",
    "JinaReaderTools",
    "YouTubeTools",
    "SQLTools",
    "ShellTools",
    "PythonTools",
]

from .ext import zyxModuleLoader


class Toolkit(zyxModuleLoader):
    pass


Toolkit.init("phi.tools.toolkit", "Toolkit")


class Tool(zyxModuleLoader):
    pass


Tool.init("phi.tools.tool", "Tool")


class Calculator(zyxModuleLoader):
    pass


Calculator.init("phi.tools.calculator", "Calculator")


class ApifyTools(zyxModuleLoader):
    pass


ApifyTools.init("phi.tools.apify", "ApifyTools")


class DuckDuckGo(zyxModuleLoader):
    pass


DuckDuckGo.init("phi.tools.duckduckgo", "DuckDuckGo")


class CsvTools(zyxModuleLoader):
    pass


CsvTools.init("phi.tools.csv_tools", "CsvTools")


class ArxivToolkit(zyxModuleLoader):
    pass


ArxivToolkit.init("phi.tools.arxiv_toolkit", "ArxivToolkit")


class WikipediaTools(zyxModuleLoader):
    pass


WikipediaTools.init("phi.tools.wikipedia", "WikipediaTools")


class WebsiteTools(zyxModuleLoader):
    pass


WebsiteTools.init("phi.tools.website", "WebsiteTools")


class JinaReaderTools(zyxModuleLoader):
    pass


JinaReaderTools.init("phi.tools.jina_tools", "JinaReaderTools")


class YouTubeTools(zyxModuleLoader):
    pass


YouTubeTools.init("phi.tools.youtube_tools", "YouTubeTools")


class SQLTools(zyxModuleLoader):
    pass


SQLTools.init("phi.tools.sql", "SQLTools")


class ShellTools(zyxModuleLoader):
    pass


ShellTools.init("phi.tools.shell", "ShellTools")


class PythonTools(zyxModuleLoader):
    pass


PythonTools.init("phi.tools.python", "PythonTools")
