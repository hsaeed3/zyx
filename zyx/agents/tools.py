__all__ = [
    "BrowserbaseLoadTool",
    "CodeDocsSearchTool",
    "CodeInterpreterTool",
    "ComposioTool",
    "CSVSearchTool",
    "DallETool",
    "DirectoryReadTool",
    "DirectorySearchTool",
    "DOCXSearchTool",
    "FileReadToolSchema",
    "FileWriterTool",
    "FileWriterToolInput",
    "GithubSearchTool",
    "MySQLSearchTool",
    "PDFTextWritingTool",
    "ScrapeElementFromWebsiteTool",
    "SerperDevTool",
    "TXTSearchTool",
    "VisionTool",
    "WebsiteSearchTool",
    "YoutubeChannelSearchTool",
    "YoutubeVideoSearchTool",
]

from ..core.main import zyxModuleLoader


class BrowserbaseLoadTool(zyxModuleLoader):
    pass


BrowserbaseLoadTool.init(
    "crewai_tools.tools.browserbase_load_tool.browserbase_load_tool",
    "BrowserbaseLoadTool",
)


class CodeDocsSearchTool(zyxModuleLoader):
    pass


CodeDocsSearchTool.init(
    "crewai_tools.tools.code_docs_search_tool.code_docs_search_tool",
    "CodeDocsSearchTool",
)


class CodeInterpreterTool(zyxModuleLoader):
    pass


CodeInterpreterTool.init(
    "crewai_tools.tools.code_interpreter_tool.code_interpreter_tool",
    "CodeInterpreterTool",
)


class ComposioTool(zyxModuleLoader):
    pass


ComposioTool.init("crewai_tools.tools.composio_tool.composio_tool", "ComposioTool")


class CSVSearchTool(zyxModuleLoader):
    pass


CSVSearchTool.init(
    "crewai_tools.tools.csv_search_tool.csv_search_tool", "CSVSearchTool"
)


class DallETool(zyxModuleLoader):
    pass


DallETool.init("crewai_tools.tools.dalle_tool.dalle_tool", "DallETool")


class DirectoryReadTool(zyxModuleLoader):
    pass


DirectoryReadTool.init(
    "crewai_tools.tools.directory_read_tool.directory_read_tool", "DirectoryReadTool"
)


class DirectorySearchTool(zyxModuleLoader):
    pass


DirectorySearchTool.init(
    "crewai_tools.tools.directory_search_tool.directory_search_tool",
    "DirectorySearchTool",
)


class DOCXSearchTool(zyxModuleLoader):
    pass


DOCXSearchTool.init(
    "crewai_tools.tools.docx_search_tool.docx_search_tool", "DOCXSearchTool"
)


class DuckDuckGoSearchRun(zyxModuleLoader):
    pass


DuckDuckGoSearchRun.init("langchain.tools.ddg_search", "DuckDuckGoSearchRun")


class FileReadToolSchema(zyxModuleLoader):
    pass


FileReadToolSchema.init(
    "crewai_tools.tools.file_read_tool.file_read_tool", "FileReadToolSchema"
)


class FileWriterTool(zyxModuleLoader):
    pass


FileWriterTool.init(
    "crewai_tools.tools.file_writer_tool.file_writer_tool", "FileWriterTool"
)


class FileWriterToolInput(zyxModuleLoader):
    pass


FileWriterToolInput.init(
    "crewai_tools.tools.file_writer_tool.file_writer_tool", "FileWriterToolInput"
)


class GithubSearchTool(zyxModuleLoader):
    pass


GithubSearchTool.init(
    "crewai_tools.tools.github_search_tool.github_search_tool", "GithubSearchTool"
)


class MySQLSearchTool(zyxModuleLoader):
    pass


MySQLSearchTool.init(
    "crewai_tools.tools.mysql_search_tool.mysql_search_tool", "MySQLSearchTool"
)


class PDFTextWritingTool(zyxModuleLoader):
    pass


PDFTextWritingTool.init(
    "crewai_tools.tools.pdf_text_writing_tool.pdf_text_writing_tool",
    "PDFTextWritingTool",
)


class ScrapeElementFromWebsiteTool(zyxModuleLoader):
    pass


ScrapeElementFromWebsiteTool.init(
    "crewai_tools.tools.scrape_element_from_website.scrape_element_from_website",
    "ScrapeElementFromWebsiteTool",
)


class SerperDevTool(zyxModuleLoader):
    pass


SerperDevTool.init(
    "crewai_tools.tools.serper_dev_tool.serper_dev_tool", "SerperDevTool"
)


class TXTSearchTool(zyxModuleLoader):
    pass


TXTSearchTool.init(
    "crewai_tools.tools.txt_search_tool.txt_search_tool", "TXTSearchTool"
)


class VisionTool(zyxModuleLoader):
    pass


VisionTool.init("crewai_tools.tools.vision_tool.vision_tool", "VisionTool")


class WebsiteSearchTool(zyxModuleLoader):
    pass


WebsiteSearchTool.init(
    "crewai_tools.tools.website_search.website_search_tool", "WebsiteSearchTool"
)


class YoutubeChannelSearchTool(zyxModuleLoader):
    pass


YoutubeChannelSearchTool.init(
    "crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool",
    "YoutubeChannelSearchTool",
)


class YoutubeVideoSearchTool(zyxModuleLoader):
    pass


YoutubeVideoSearchTool.init(
    "crewai_tools.tools.youtube_video_search_tool.youtube_video_search_tool",
    "YoutubeVideoSearchTool",
)
