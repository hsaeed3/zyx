# zyx.resources.types.completions.arguments
# completion types (args & response)


from __future__ import annotations


__all__ = [
    "Completion",
    "CompletionArguments"
]


from .instructor import InstructorMode
from .messages import Message
from ..tools.tool import ToolType, Tool

from pydantic import BaseModel, ConfigDict
from typing import Any, Iterable, Generator, Optional, Union, Dict, List, Literal, Type, Sequence
from typing_extensions import TypedDict

# openai type imports
from openai.types.chat.completion_create_params import (
    Function, FunctionCall, ResponseFormat,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_modality import ChatCompletionModality
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam

from openai._streaming import Stream


# process
# NOTE only chain of thought is implemented
# TODO add more processes later
Process = Literal["chain_of_thought"]


# Chat Models
# predefined models
ChatModel = Literal[
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


# IMPORTANT TYPE
# completion response type
Completion = Union[
    # standard openai completion types
    ChatCompletion, Generator[ChatCompletionChunk, None, None],
    # instructor type
    Type[BaseModel],

    Stream[ChatCompletionChunk]
]


# post arguments
# SENT AT RUNTIME
class CompletionPOST(TypedDict):

    model : str
    messages : List[ChatCompletionMessageParam]

    # openai pruned
    audio: Optional[ChatCompletionAudioParam] = None
    frequency_penalty: Optional[float] = None
    function_call: FunctionCall = None
    functions: Iterable[Function] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    modalities: Optional[List[ChatCompletionModality]] = None
    n: Optional[int] = None
    parallel_tool_calls: bool = False
    presence_penalty: Optional[float] = None
    response_format: ResponseFormat = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default"]] = None
    stop: Union[Optional[str], List[str]] = None
    store: Optional[bool] = None
    stream: Optional[Literal[False]] | Literal[True] = None
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None
    temperature: Optional[float] = None
    tool_choice: ChatCompletionToolChoiceOptionParam = None
    tools: Iterable[ChatCompletionToolParam] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: Optional[str] = None


class CompletionInstructorPOST(TypedDict):

    # instructor response model
    response_model : Type[BaseModel]


# completion arguments
class CompletionArguments(BaseModel):

    # arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # core completions arguments
    # !! REQUIRED 
    messages: Union[str, List[Dict[str, Any]]]
    model: Union[str, ChatModel]

    # instructor arguments
    mode : Optional[InstructorMode] = None
    # (zyx uses instructor as a default for structured output)
    # the "response_format" arg is swapped for "response_model"
    response_model : Optional[Type[BaseModel]] = None

    # run_tools
    # automatically executes applicable & relevant tools
    # !! defaults to True
    run_tools : Optional[bool] = True

    # chat (launches cli chatbot)
    chat : Optional[bool] = None

    # progress_bar (rich.progress.Progress)
    progress_bar : Optional[bool] = True

    # process (used for response_model processing & tool output)
    process : Optional[Process] = None

    # POST request args
    # (openai pruned)
    audio: Optional[ChatCompletionAudioParam] = None
    frequency_penalty: Optional[float] = None
    function_call: Optional[FunctionCall] = None
    functions: Optional[Iterable[Function]] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    modalities: Optional[List[ChatCompletionModality]] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = False
    presence_penalty: Optional[float] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default"]] = None
    stop: Union[Optional[str], List[str]] = None
    store: Optional[bool] = None
    stream: Optional[Literal[False]] | Literal[True] = None
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[List[Union[str, ToolType]]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    

    def format_messages(
        self,
        messages: Optional[Union[str, Sequence[Message]]] = None,
        verbose: bool = False,
        type: Literal["user", "system", "assistant"] = "user",
    ) -> List[Dict[str, str]]:
        """Formats the messages into a list of dictionaries.
        
        Args:
            messages: The messages to format.
            verbose: Whether to print verbose output.
            type: The type of message to format.
        """
        messages = messages or self.messages

        if isinstance(messages, str):
            if verbose:
                print("Converting string to message format.")
            formatted = [{"role": type, "content": messages}]
        elif isinstance(messages, Sequence) and all(isinstance(m, (dict, Message)) for m in messages):
            if verbose:
                print("Messages are in the correct format.")
            formatted = [m.dict() if isinstance(m, Message) else m for m in messages]
        else:
            raise ValueError("Invalid message format")

        self.messages = formatted
        return formatted


    def convert_to_image_message(
        self,
        message: Union[str, Dict[str, str]],
        image: str
    ) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]:
        import base64
        from pathlib import Path

        if Path(image).is_file():
            with open(image, "rb") as image_file:
                image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
        elif not (image.startswith("http") or image.startswith("https") or image.startswith("data:image")):
            raise ValueError("Invalid image format. Must be a file path, URL, or base64 encoded string.")

        content = message if isinstance(message, str) else message.get('content', '')

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": image}}
            ]
        }


    def does_system_prompt_exist(self) -> bool:
        """Checks if a system prompt exists in the messages."""
        return any(message.get("role") == "system" for message in self.messages)


    def swap_system_prompt(
        self,
        system_prompt: Dict[str, str],
        messages: Optional[Union[str, Sequence[Message]]] = None
    ) -> List[Dict[str, str]]:
        """Swaps the system prompt with the given system_prompt."""
        messages = self.format_messages(messages or self.messages)
        
        system_messages = [i for i, msg in enumerate(messages) if msg.get("role") == "system"]
        if system_messages:
            messages[system_messages[0]] = system_prompt
            for i in reversed(system_messages[1:]):
                messages.pop(i)
        else:
            messages.insert(0, system_prompt)

        self.messages = messages
        return messages


    def repair_messages(self, verbose: bool = False) -> List[Dict[str, str]]:
        """Repairs the messages by performing quick logic steps."""
        messages = self.messages

        # flatten nested lists
        if any(isinstance(message, list) for message in messages):
            messages = [item for sublist in messages for item in (sublist if isinstance(sublist, list) else [sublist])]
            if verbose:
                print("Detected nested lists and flattened the list.")

        # validate message structure
        valid_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            # ensure required fields exist
            if "role" not in message:
                continue
                
            # normalize tool response roles
            if message["role"] == "tool_response":
                message["role"] = "tool"
                
            # ensure content exists
            if "content" not in message:
                message["content"] = ""
                
            valid_messages.append(message)
            
        messages = valid_messages

        # validate conversation flow
        for i in range(len(messages) - 1):
            curr_role = messages[i].get("role")
            next_role = messages[i + 1].get("role")
            
            # handle assistant -> user flow
            if curr_role == "assistant" and next_role not in ["user", "tool"]:
                messages.insert(i + 1, {"role": "user", "content": ""})
                if verbose:
                    print(f"Inserted empty user message at position {i + 1}")
                    
            # handle user -> assistant flow    
            elif curr_role == "user" and next_role not in ["assistant", "tool"]:
                messages.insert(i + 1, {"role": "assistant", "content": ""})
                if verbose:
                    print(f"Inserted empty assistant message at position {i + 1}")

        self.messages = messages
        return messages


    def add_messages(
        self,
        inputs: Union[str, Sequence[Message], Dict[str, str], Dict],
        type: Literal["user", "system", "assistant", "tool_call", "tool_response"] = "user",
        verbose: bool = False,
    ) -> List[Dict[str, str]]:
        """Adds a message or messages to the thread."""
        if isinstance(inputs, str):
            new_messages = [{"role": type, "content": inputs}]
        elif isinstance(inputs, dict):
            if "choices" in inputs:
                for choice in inputs["choices"]:
                    if "tool_calls" in choice["message"]:
                        new_messages = [{
                            "role": "tool",
                            "content": choice["message"]["tool_calls"][0]["function"]["arguments"],
                            "tool_call_id": choice["message"]["tool_calls"][0]["id"]
                        }]
                    else:
                        new_messages = [choice["message"]]
            else:
                new_messages = [inputs]
        elif isinstance(inputs, Sequence):
            new_messages = [
                item if isinstance(item, dict) else item.dict() if isinstance(item, Message) else {"role": type, "content": str(item)}
                for item in inputs
            ]
        else:
            raise ValueError("Invalid input format")

        # Ensure tool_response is handled as type: tool
        for message in new_messages:
            if message.get("role") == "tool_response":
                message["role"] = "tool"

        self.messages.extend(new_messages)
        return self.repair_messages(verbose)
    

    def add_tool_execution_output(
            self,
            messages : List[Dict[str, Any]],
            id : str,
            output : Any
    ) -> List[Dict[str, Any]]:
        """Inspects tool call and adds the tool response message to the thread."""

        tool_response_message = {
            "role": "tool",
            "content": str(output),
            "tool_call_id": id
        }

        messages.append(tool_response_message)

        return messages
        

    def message_sequence_to_openai_param(self, messages: Sequence[Message]) -> List[ChatCompletionMessageParam]:
        """Converts a sequence of messages to a list of ChatCompletionMessageParam."""
        return [message.dict() if hasattr(message, 'dict') else message for message in messages]


    def add_tools(self, tools : List[Tool]) -> None:
        """Adds tools to the completion arguments."""
        
        tool_dicts = []

        for tool in tools:
            tool_dicts.append(
                tool.formatted_function
            )

        self.tools = tool_dicts


    def model_dump_POST(self, instructor: bool = False, response_model: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """Builds a request dict from the model Formats into CompletionPOSTArguments & returns dict."""
        # build post args model
        model_data = self.model_dump()

        # Ensure all optional fields are set to their default values if they are None
        default_values = {
            "function_call": None,
            "functions": None,
            "response_format": None,
            "tool_choice": None,
            "tools": None,
            "response_model": None if not instructor else self.response_model
        }

        for key, default in default_values.items():
            if model_data.get(key) is None:
                model_data[key] = default

        # Handle messages field specifically
        if isinstance(model_data.get('messages'), list):
            model_data['messages'] = [
                message.dict() if hasattr(message, 'dict') else message
                for message in model_data['messages']
            ]

        # Exclude non-request args
        excluded_keys = {"mode", "chat", "progress_bar", "process", "run_tools"}
        request_data = {k: v for k, v in model_data.items() if k not in excluded_keys and v is not None}

        if instructor:
            request_data["response_model"] = self.response_model
        else:
            # ensure response_model is not included
            request_data.pop("response_model", None)

        if not "tools" in request_data:
            # drop parallel_tool_calls, tool_choice, and tools
            request_data.pop("parallel_tool_calls", None)
            request_data.pop("tool_choice", None)
            request_data.pop("tools", None)

        return request_data



if __name__ == "__main__":

    # test model init
    model = CompletionArguments(model="gpt-4o", messages="Hello, how are you?")

    # test format_messages
    model.format_messages()

    print(model)

    print(model.messages)

    # test add_messages with string input
    model.add_messages("I'm fine, thank you!", type="user", verbose=True)
    print(model.messages)

    # test add_messages with dict input
    model.add_messages({"role": "assistant", "content": "That's great to hear!"}, verbose=True)
    print(model.messages)

    # test add_messages with sequence input
    model.add_messages([{"role": "user", "content": "What can you do?"}, {"role": "assistant", "content": "I can help you with various tasks."}], verbose=True)
    print(model.messages)

    # test add_messages with complex dict input
    model.add_messages({"choices": [{"message": {"tool_calls": [{"function": {"arguments": "arg1, arg2"}, "id": "tool1"}]}}]}, verbose=True)
    print(model.messages)

    # test repair_messages
    model.repair_messages(verbose=True)
    print(model.messages)

    # test model_dump_POST
    print(model.model_dump_POST(instructor=True))
