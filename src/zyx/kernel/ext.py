# zyx ==============================================================================

from zyx.ext._loader import UtilLazyLoader


class Kernel(UtilLazyLoader):
    pass


Kernel.init("semantic_kernel.kernel", "Kernel")


class OllamaChatCompletion(UtilLazyLoader):
    pass


OllamaChatCompletion.init(
    "semantic_kernel.connectors.ai.ollama", "OllamaChatCompletion"
)


class OllamaTextEmbedding(UtilLazyLoader):
    pass


OllamaTextEmbedding.init("semantic_kernel.connectors.ai.ollama", "OllamaTextEmbedding")


class OllamaPromptExecutionSettings(UtilLazyLoader):
    pass


OllamaPromptExecutionSettings.init(
    "semantic_kernel.connectors.ai.ollama", "OllamaPromptExecutionSettings"
)


class OpenAIChatCompletion(UtilLazyLoader):
    pass


OpenAIChatCompletion.init(
    "semantic_kernel.connectors.ai.open_ai", "OpenAIChatCompletion"
)


class OpenAITextEmbedding(UtilLazyLoader):
    pass


OpenAITextEmbedding.init("semantic_kernel.connectors.ai.open_ai", "OpenAITextEmbedding")


class OpenAIPromptExecutionSettings(UtilLazyLoader):
    pass


OpenAIPromptExecutionSettings.init(
    "semantic_kernel.connectors.ai.open_ai", "OpenAIPromptExecutionSettings"
)


class ChatHistory(UtilLazyLoader):
    pass


ChatHistory.init("semantic_kernel.contents.chat_history", "ChatHistory")


class ChatMessageContent(UtilLazyLoader):
    pass


ChatMessageContent.init(
    "semantic_kernel.contents.chat_message_content", "ChatMessageContent"
)


class FunctionCallContent(UtilLazyLoader):
    pass


FunctionCallContent.init(
    "semantic_kernel.contents.function_call_content", "FunctionCallContent"
)


class StreamingChatMessageContent(UtilLazyLoader):
    pass


StreamingChatMessageContent.init(
    "semantic_kernel.contents.streaming_chat_message_content",
    "StreamingChatMessageContent",
)


class KernelFunction(UtilLazyLoader):
    pass


KernelFunction.init("semantic_kernel.functions.kernel_function", "KernelFunction")


class KernelArguments(UtilLazyLoader):
    pass


KernelArguments.init("semantic_kernel.functions.kernel_arguments", "KernelArguments")


class kernel_function(UtilLazyLoader):
    pass


kernel_function.init(
    "semantic_kernel.functions.kernel_function_decorator", "kernel_function"
)


class KernelFunctionExtension(UtilLazyLoader):
    pass


KernelFunctionExtension.init(
    "semantic_kernel.functions.kernel_function_extension", "KernelFunctionExtension"
)


class ChatCompletionAgent(UtilLazyLoader):
    pass


ChatCompletionAgent.init(
    "semantic_kernel.agents.chat_completion_agent", "ChatCompletionAgent"
)


class Agent(UtilLazyLoader):
    pass


Agent.init("semantic_kernel.agents.agent", "Agent")


class FunctionChoiceBehavior(UtilLazyLoader):
    pass


FunctionChoiceBehavior.init(
    "semantic_kernel.connectors.ai.function_choice_behavior", "FunctionChoiceBehavior"
)


class PromptExecutionSettings(UtilLazyLoader):
    pass


PromptExecutionSettings.init(
    "semantic_kernel.connectors.ai.prompt_execution_settings", "PromptExecutionSettings"
)


class auto_function_invocation_context(UtilLazyLoader):
    pass


auto_function_invocation_context.init(
    "semantic_kernel.filters.auto_function_invocation",
    "auto_function_invocation_context",
)


class DocumentLoader(UtilLazyLoader):
    pass


DocumentLoader.init(
    "semantic_kernel.connectors.utils.document_loader", "DocumentLoader"
)


class ChatCompletionClientBase(UtilLazyLoader):
    pass


ChatCompletionClientBase.init(
    "semantic_kernel.connectors.ai.chat_completion_client_base",
    "ChatCompletionClientBase",
)


class QdrantMemoryStore(UtilLazyLoader):
    pass


QdrantMemoryStore.init("semantic_kernel.connectors.memory.qdrant", "QdrantMemoryStore")


class BingConnector(UtilLazyLoader):
    pass


BingConnector.init(
    "semantic_kernel.connectors.search_engine.bing_connector", "BingConnector"
)


class GoogleConnector(UtilLazyLoader):
    pass


GoogleConnector.init(
    "semantic_kernel.connectors.search_engine.google_connector", "GoogleConnector"
)


class StreamingContentMixin(UtilLazyLoader):
    pass


StreamingContentMixin.init(
    "semantic_kernel.contents.streaming_content_mixin", "StreamingContentMixin"
)


class KernelContent(UtilLazyLoader):
    pass


KernelContent.init("semantic_kernel.contents.kernel_content", "KernelContent")
