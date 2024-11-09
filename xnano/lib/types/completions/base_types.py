# base types for completions

from typing import Literal, TypeAlias, Type, Union


Completions: TypeAlias = "xnano.completions.Completions"


# chat model literals
CompletionsChatModels = Literal[
    "o1-preview", "o1-mini",
    "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    "anthropic/claude-3-5-sonnet-20240620", "anthropic/claude-2.1", "anthropic/claude-2", "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-sonnet-latest",
    "anthropic/claude-3-haiku-20240307", "anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229",
    "ollama/llama3.2", "ollama/llama3.2:3b", "ollama/llama3.2:1b",
    "ollama/llama3.1", "ollama/llama3.1:8b", "ollama/llama3.1:70b",
    "ollama/llama3", "ollama/llama3:8b", "ollama/llama3:70b",
    "ollama/mistral-nemo",
    "ollama/nemotron-mini",
    "ollama/llava",
    "ollama/mistral", "ollama/mistral:7b", "ollama/mistral:7b:instruct",
    "ollama/mixtral", "ollama/mixtral:8x7b", "ollama/mixtral:8x7b:instruct",
    "ollama/gemma2", "ollama/gemma2:9b", "ollama/gemma2:27b",
    "ollama/phi3.5",
    "ollama/qwen2.5", "ollama/qwen2.5:0.5b", "ollama/qwen2.5:1.5b", "ollama/qwen2.5:3b",
    "ollama/qwen2.5:7b", "ollama/qwen2.5:14b", "ollama/qwen2.5:32b", "ollama/qwen2.5:72b",
    "ollama/nuextract",
    "ollama/granite3-moe:1b", "ollama/granite3-moe:3b", "ollama/granite3-dense:2b", "ollama/granite3-dense:8b",
    "ollama/solar-pro", "ollama/llama-guard3:1b", "ollama/llama-guard3:8b",
]


# chat model
CompletionsChatModel = Union[
    str,
    CompletionsChatModels
]


# instructor mode
CompletionsInstructorMode = Literal[
    "function_call", "parallel_tool_call", "tool_call", "mistral_tools",
    "json_mode", "json_o1", "markdown_json_mode", "json_schema_mode",
    "anthropic_tools", "anthropic_json", "cohere_tools", "vertexai_tools",
    "vertexai_json", "gemini_json", "gemini_tools", "json_object", "tools_strict",
]