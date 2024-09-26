__all__ = [
    "convert_to_openai_tool",
    "load_docs",
    "simple_text_loader",
    "extract_metadata",
    "hash_documents",
    "extract_keywords",
    "summarize",
    "export_documents_to_json",
    "import_documents_from_json",
    "generate_document_report",
    "format_messages",
    "does_system_prompt_exist",
    "swap_system_prompt",
    "repair_messages",
    "add_messages",
    "logger",
    "tqdm",
]


from .._loader import loader


class convert_to_openai_tool(loader):
    pass


convert_to_openai_tool.init(
    "zyx.lib.utils.convert_to_openai_tool", "convert_to_openai_tool"
)


class load_docs(loader):
    pass


load_docs.init("zyx.lib.utils.data", "load_docs")


class simple_text_loader(loader):
    pass


simple_text_loader.init("zyx.lib.utils.data", "simple_text_loader")


class extract_metadata(loader):
    pass


extract_metadata.init("zyx.lib.utils.data", "extract_metadata")


class hash_documents(loader):
    pass


hash_documents.init("zyx.lib.utils.data", "hash_documents")


class extract_keywords(loader):
    pass


extract_keywords.init("zyx.lib.utils.data", "extract_keywords")


class summarize(loader):
    pass


summarize.init("zyx.lib.utils.data", "summarize")


class export_documents_to_json(loader):
    pass


export_documents_to_json.init("zyx.lib.utils.data", "export_documents_to_json")


class import_documents_from_json(loader):
    pass


import_documents_from_json.init("zyx.lib.utils.data", "import_documents_from_json")


class generate_document_report(loader):
    pass


generate_document_report.init("zyx.lib.utils.data", "generate_document_report")


class format_messages(loader):
    pass


format_messages.init("zyx.lib.utils.messages", "format_messages")


class does_system_prompt_exist(loader):
    pass


does_system_prompt_exist.init("zyx.lib.utils.messages", "does_system_prompt_exist")


class swap_system_prompt(loader):
    pass


swap_system_prompt.init("zyx.lib.utils.messages", "swap_system_prompt")


class repair_messages(loader):
    pass


repair_messages.init("zyx.lib.utils.messages", "repair_messages")


class add_messages(loader):
    pass


add_messages.init("zyx.lib.utils.messages", "add_messages")


class logger(loader):
    pass


logger.init("loguru", "logger")


class tqdm(loader):
    pass


tqdm.init("tqdm", "tqdm")
