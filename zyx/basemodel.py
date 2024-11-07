# zyx.basemodel
# pydantic basemodel llm extension

__all__ = [
    "BaseModel", "Field"
]

from .completions import completion, acompletion
from .lib.console import console
from .lib.exception import ZYXException

import httpx
from .types.completions._openai import ChatCompletionModality, ChatCompletionPredictionContentParam, ChatCompletionAudioParam
from .types.completions.completion_chat_model_param import CompletionChatModelParam
from .types.completions.completion_context import CompletionContext
from .types.completions.completion_instructor_mode import CompletionInstructorMode
from .types.completions.completion_message_param import CompletionMessageParam
from .types.completions.completion_response import CompletionResponse
from .types.completions.completion_response_model_param import CompletionResponseModelParam
from .types.completions.completion_tool_choice_param import CompletionToolChoiceParam
from .types.completions.completion_tool_type import CompletionToolType

from textwrap import dedent
from pydantic import BaseModel as PydanticBaseModel, Field as Field
from typing import Any, TypeVar, Generic, Type, Optional, List, Union
import json


# typevar for basemodel type instances
T = TypeVar("T", bound=PydanticBaseModel)


# Descriptor class for fields to add LLM capabilities
class FieldDescriptor:
    """
    Descriptor to add extended functionality to fields.
    """
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.value

    def __set__(self, instance, value):
        self.value = value

    # Custom method to print field value
    def field_print(self):
        print(f"Field value: {self.value}")

    # Custom method to return JSON serializable value
    def to_json_compatible(self):
        if hasattr(self.value, "dict"):
            return self.value.dict()
        return self.value


# BaseModel extension with enhanced field capabilities
class BaseModel(PydanticBaseModel):
    """
    zyx Pydantic BaseModel extension.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache fields using FieldDescriptor for extended capabilities
        for field_name, field_value in self.__dict__.items():
            if field_name in self.__annotations__:
                super().__setattr__(field_name, FieldDescriptor(field_value))

    def __setattr__(self, name, value):
        # Wrap value with FieldDescriptor if it is a field of this model
        if name in self.__annotations__:
            value = FieldDescriptor(value)
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        # Ensure fields are returned as FieldDescriptor objects if they are defined as annotations
        if name in super().__getattribute__('__annotations__'):
            value = super().__getattribute__('__dict__').get(name, None)
            if isinstance(value, FieldDescriptor):
                return value
            return FieldDescriptor(value) if value is not None else value
        return super().__getattribute__(name)

    def model_dump_json(self, **kwargs):
        # Override to provide JSON serialization that handles FieldDescriptor
        serializable_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, FieldDescriptor):
                serializable_dict[field_name] = field_value.to_json_compatible()
            else:
                serializable_dict[field_name] = field_value
        return json.dumps(serializable_dict, **kwargs)

    # -------------------------------------------------------------------------
    # llm methods
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # completion -- for RAG/context based completions
    # -------------------------------------------------------------------------

    # NOTE::
    # all extension methods are prefixed with model_
    # this helps integrate with pydantic's default methods & allow normal field access

    @classmethod
    def model_completion(
        cls_or_self : Type[T] | T,
        messages : CompletionMessageParam,
        model : CompletionChatModelParam = "gpt-4o-mini",
        context : Optional[CompletionContext] = None,
        mode : Optional[CompletionInstructorMode] = None,
        response_model : Optional[CompletionResponseModelParam] = None,
        response_format : Optional[CompletionResponseModelParam] = None,
        tools : Optional[List[CompletionToolType]] = None,
        run_tools : Optional[bool] = None,
        tool_choice : Optional[CompletionToolChoiceParam] = None,
        parallel_tool_calls : Optional[bool] = None,
        api_key : Optional[str] = None,
        base_url : Optional[str] = None,
        organization : Optional[str] = None,
        n : Optional[int] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        modalities: Optional[List[ChatCompletionModality]] = None,
        prediction: Optional[ChatCompletionPredictionContentParam] = None,
        audio: Optional[ChatCompletionAudioParam] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        api_version: Optional[str] = None,
        model_list: Optional[list] = None, 
        stream : Optional[bool] = None,
        loader : Optional[bool] = True,
        verbose : Optional[bool] = None
    ) -> CompletionResponse:
        """
        Returns an LLM completion response using the pydantic model schema or instance as context.
        """

        # determine if instance or class
        is_instance = not isinstance(cls_or_self, type)

        # if not an instantiated model, build context using schema
        try:
            if not is_instance:
                model_name = cls_or_self.__class__.__name__ if isinstance(cls_or_self, BaseModel) else cls_or_self.__name__
                model_context = f"""
                You are an expert at building & conversing about {model_name} instances.
                
                # Instructions:
                - Answer questions or accurately respond to queries about the {model_name} class.

                # Schema:
                {cls_or_self.model_json_schema()}
                """
            else:
                model_name = type(cls_or_self).__name__
                model_context = f"""
                You are an expert at building & conversing about {model_name} instances.

                # Instructions:
                - Answer questions or accurately respond to queries about the {model_name} instance.

                # Schema:
                {cls_or_self.model_json_schema()}

                The current instance of the {model_name} is:
                {cls_or_self.model_dump_json()}
                """
        except Exception as e:
            raise ZYXException(f"Error building model context: Unable to serialize unknown type: {e}")
        
        if context is not None:
            context = dedent(f"""
            {context}

            -------------------------------------------------------------------------
            
            {model_context}
            """)
        else:
            context = model_context

        if loader:
            with console.progress(
                f"Generating completion for [bold]{model_name}[/bold]..."
            ) as progress:
                
                response = completion(
                    messages = messages, model = model, context = context, 
                    response_model = response_model, response_format = response_format,
                    tools = tools, run_tools = run_tools, tool_choice = tool_choice,
                    parallel_tool_calls = parallel_tool_calls, api_key = api_key,
                    base_url = base_url, organization = organization, n = n,
                    timeout = timeout, temperature = temperature, top_p = top_p,
                    stream_options = stream_options, stop = stop,
                    max_completion_tokens = max_completion_tokens, max_tokens = max_tokens,
                    modalities = modalities, prediction = prediction, audio = audio,
                    presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
                    logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs,
                    top_logprobs = top_logprobs, deployment_id = deployment_id,
                    extra_headers = extra_headers, functions = functions, function_call = function_call,
                    api_version = api_version, model_list = model_list, stream = stream, verbose = verbose
                )

                return response

        else:

            return completion(
                    messages = messages, model = model, context = context, 
                    response_model = response_model, response_format = response_format,
                    tools = tools, run_tools = run_tools, tool_choice = tool_choice,
                    parallel_tool_calls = parallel_tool_calls, api_key = api_key,
                    base_url = base_url, organization = organization, n = n,
                    timeout = timeout, temperature = temperature, top_p = top_p,
                    stream_options = stream_options, stop = stop,
                    max_completion_tokens = max_completion_tokens, max_tokens = max_tokens,
                    modalities = modalities, prediction = prediction, audio = audio,
                    presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
                    logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs,
                    top_logprobs = top_logprobs, deployment_id = deployment_id,
                    extra_headers = extra_headers, functions = functions, function_call = function_call,
                    api_version = api_version, model_list = model_list, stream = stream, verbose = verbose
                )  
        

    @classmethod
    async def model_acompletion(
        cls_or_self : Type[T] | T,
        messages : CompletionMessageParam,
        model : CompletionChatModelParam = "gpt-4o-mini",
        context : Optional[CompletionContext] = None,
        mode : Optional[CompletionInstructorMode] = None,
        response_model : Optional[CompletionResponseModelParam] = None,
        response_format : Optional[CompletionResponseModelParam] = None,
        tools : Optional[List[CompletionToolType]] = None,
        run_tools : Optional[bool] = None,
        tool_choice : Optional[CompletionToolChoiceParam] = None,
        parallel_tool_calls : Optional[bool] = None,
        api_key : Optional[str] = None,
        base_url : Optional[str] = None,
        organization : Optional[str] = None,
        n : Optional[int] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        modalities: Optional[List[ChatCompletionModality]] = None,
        prediction: Optional[ChatCompletionPredictionContentParam] = None,
        audio: Optional[ChatCompletionAudioParam] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        api_version: Optional[str] = None,
        model_list: Optional[list] = None, 
        stream : Optional[bool] = None,
        loader : Optional[bool] = True,
        verbose : Optional[bool] = None
    ) -> CompletionResponse:
        """
        Returns an LLM completion response using the pydantic model schema or instance as context.
        """

        # determine if instance or class
        is_instance = not isinstance(cls_or_self, type)

        # if not an instantiated model, build context using schema
        try:
            if not is_instance:
                model_name = cls_or_self.__class__.__name__ if isinstance(cls_or_self, BaseModel) else cls_or_self.__name__
                model_context = f"""
                You are an expert at building & conversing about {model_name} instances.
                
                # Instructions:
                - Answer questions or accurately respond to queries about the {model_name} class.

                # Schema:
                {cls_or_self.model_json_schema()}
                """
            else:
                model_name = type(cls_or_self).__name__
                model_context = f"""
                You are an expert at building & conversing about {model_name} instances.

                # Instructions:
                - Answer questions or accurately respond to queries about the {model_name} instance.

                # Schema:
                {cls_or_self.model_json_schema()}

                The current instance of the {model_name} is:
                {cls_or_self.model_dump_json()}
                """
        except Exception as e:
            raise ZYXException(f"Error building model context: Unable to serialize unknown type: {e}")
        
        if context is not None:
            context = dedent(f"""
            {context}

            -------------------------------------------------------------------------
            
            {model_context}
            """)
        else:
            context = model_context

        if loader:
            with console.progress(
                f"Generating completion for [bold]{model_name}[/bold]..."
            ) as progress:
                
                response = await acompletion(
                    messages = messages, model = model, context = context, 
                    response_model = response_model, response_format = response_format,
                    tools = tools, run_tools = run_tools, tool_choice = tool_choice,
                    parallel_tool_calls = parallel_tool_calls, api_key = api_key,
                    base_url = base_url, organization = organization, n = n,
                    timeout = timeout, temperature = temperature, top_p = top_p,
                    stream_options = stream_options, stop = stop,
                    max_completion_tokens = max_completion_tokens, max_tokens = max_tokens,
                    modalities = modalities, prediction = prediction, audio = audio,
                    presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
                    logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs,
                    top_logprobs = top_logprobs, deployment_id = deployment_id,
                    extra_headers = extra_headers, functions = functions, function_call = function_call,
                    api_version = api_version, model_list = model_list, stream = stream, verbose = verbose
                )

                return response

        else:

            return await acompletion(
                    messages = messages, model = model, context = context, 
                    response_model = response_model, response_format = response_format,
                    tools = tools, run_tools = run_tools, tool_choice = tool_choice,
                    parallel_tool_calls = parallel_tool_calls, api_key = api_key,
                    base_url = base_url, organization = organization, n = n,
                    timeout = timeout, temperature = temperature, top_p = top_p,
                    stream_options = stream_options, stop = stop,
                    max_completion_tokens = max_completion_tokens, max_tokens = max_tokens,
                    modalities = modalities, prediction = prediction, audio = audio,
                    presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
                    logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs,
                    top_logprobs = top_logprobs, deployment_id = deployment_id,
                    extra_headers = extra_headers, functions = functions, function_call = function_call,
                    api_version = api_version, model_list = model_list, stream = stream, verbose = verbose
                )  
        

    # -------------------------------------------------------------------------
    # generator
    # -------------------------------------------------------------------------

    @classmethod
    def model_generate(
        cls_or_self : Type[T] | T,
        messages : CompletionMessageParam,
        model : CompletionChatModelParam = "gpt-4o-mini",
        context : Optional[CompletionContext] = None,
        mode : Optional[CompletionInstructorMode] = None,
        tools : Optional[List[CompletionToolType]] = None,
        run_tools : Optional[bool] = None,
        tool_choice : Optional[CompletionToolChoiceParam] = None,
        parallel_tool_calls : Optional[bool] = None,
        api_key : Optional[str] = None,
        base_url : Optional[str] = None,
        organization : Optional[str] = None,
        n : Optional[int] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        modalities: Optional[List[ChatCompletionModality]] = None,
        prediction: Optional[ChatCompletionPredictionContentParam] = None,
        audio: Optional[ChatCompletionAudioParam] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        api_version: Optional[str] = None,
        model_list: Optional[list] = None, 
        stream : Optional[bool] = None,
        loader : Optional[bool] = True,
        verbose : Optional[bool] = None
    ) -> CompletionResponse:
        
        """
        Returns an LLM completion response using the pydantic model schema or instance as context.
        """

        # determine if instance or class
        is_instance = not isinstance(cls_or_self, type)
        

if __name__ == "__main__":

    # tests
    class Sentiment(BaseModel):
        value: str
        confidence: float

    sentiment = Sentiment(value="happy", confidence=0.95)

    # Access the custom method on the field
    sentiment.value.field_print()  # Output: Field value: happy

    sentiment.value = "sad"
    sentiment.value.field_print()  # Output: Field value: sad

    print(sentiment.model_completion(messages=[{"role": "user", "content": "What is the sentiment?"}]))

    print(
        Sentiment.model_completion(messages=[{"role": "user", "content": "What is the sentiment?"}])
    )

    import asyncio
    print(
        asyncio.run(
            Sentiment.model_acompletion(messages=[{"role": "user", "content": "What is the sentiment?"}])
        )
    )
