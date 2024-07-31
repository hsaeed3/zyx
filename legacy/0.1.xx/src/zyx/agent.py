# zyx ==============================================================================

from .ext._loader import UtilLazyLoader


class tool(UtilLazyLoader):
    pass


tool.init("langchain_core.tools", "tool")


class ChatPromptTemplate(UtilLazyLoader):
    pass


ChatPromptTemplate.init("langchain_core.prompts", "ChatPromptTemplate")


class MessagesPlaceholder(UtilLazyLoader):
    pass


MessagesPlaceholder.init("langchain_core.prompts", "MessagesPlaceholder")


class BaseMessage(UtilLazyLoader):
    pass


BaseMessage.init("langchain_core.messages", "BaseMessage")


class AIMessage(UtilLazyLoader):
    pass


AIMessage.init("langchain_core.messages", "AIMessage")


class HumanMessage(UtilLazyLoader):
    pass


HumanMessage.init("langchain_core.messages", "HumanMessage")


class SystemMessage(UtilLazyLoader):
    pass


SystemMessage.init("langchain_core.messages", "SystemMessage")


class ToolMessage(UtilLazyLoader):
    pass


ToolMessage.init("langchain_core.messages", "ToolMessage")


class BaseModel(UtilLazyLoader):
    pass


BaseModel.init("langchain_core.pydantic_v1", "BaseModel")


class Field(UtilLazyLoader):
    pass


Field.init("langchain_core.pydantic_v1", "Field")


class ensure_config(UtilLazyLoader):
    pass


ensure_config.init("langchain_core.runnables.config", "ensure_config")


class ChatOpenAI(UtilLazyLoader):
    pass


ChatOpenAI.init("langchain_openai.chat_models.base", "ChatOpenAI")


class OpenAIEmbeddings(UtilLazyLoader):
    pass


OpenAIEmbeddings.init("langchain_openai.embeddings.base", "OpenAIEmbeddings")


class ChatAnthropic(UtilLazyLoader):
    pass


ChatAnthropic.init("langchain_anthropic.chat_models", "ChatAnthropic")
