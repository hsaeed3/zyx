# zyx.basemodel
# pydantic basemodel llm extension

__all__ = [
    "BaseModel", "Field"
]

from .completions import completion, acompletion
from .console import console
from .exceptions import LLXMException

import httpx
from .types.basemodel.basemodel_generation_process import BaseModelGenerationProcess
from .types.completions._openai import ChatCompletionModality, ChatCompletionPredictionContentParam, ChatCompletionAudioParam
from .types.completions.completions_chat_model import CompletionsChatModel
from .types.completions.completions_context import CompletionsContext
from .types.completions.completions_instructor_mode import CompletionsInstructorMode
from .types.completions.completions_message_type import CompletionsMessageType
from .types.completions.completions_response import CompletionsResponse
from .types.completions.completions_response_model import CompletionsResponseModel
from .types.completions.completions_tool_choice import CompletionsToolChoice
from .types.completions.completions_tool_type import CompletionsToolType

from textwrap import dedent
from pydantic import BaseModel as PydanticBaseModel, Field as Field, create_model
from typing import Any, TypeVar, Generic, Type, Optional, List, Union, overload, Self
import json


# typevar for basemodel type instances
T = TypeVar("T", bound=PydanticBaseModel)
B = TypeVar("B", bound='BaseModel')


# BaseModel extension with enhanced field capabilities
class BaseModel(PydanticBaseModel):

    # enable arbitrary types as this is still a pydantic model
    class Config:
        arbitrary_types_allowed = True

    """
    zyx Pydantic BaseModel extension.
    """

    def __init__(self, *args, **kwargs):
        # Initialize Pydantic model first
        super().__init__(*args, **kwargs)

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
        cls_or_self : Type[T] | B,
        messages : CompletionsMessageType,
        model : CompletionsChatModel = "gpt-4o-mini",
        context : Optional[CompletionsContext] = None,
        mode : Optional[CompletionsInstructorMode] = None,
        response_model : Optional[CompletionsResponseModel] = None,
        response_format : Optional[CompletionsResponseModel] = None,
        tools : Optional[List[CompletionsToolType]] = None,
        run_tools : Optional[bool] = None,
        tool_choice : Optional[CompletionsToolChoice] = None,
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
    ) -> CompletionsResponse:
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
            raise LLXMException(f"Error building model context: Unable to serialize unknown type: {e}")
        
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
                    messages = messages, model = model, context = context, mode = mode,
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
                    messages = messages, model = model, context = context, mode = mode,
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
    async def model_async_completion(
        cls_or_self : Type[T] | B,
        messages : CompletionsMessageType,
        model : CompletionsChatModel = "gpt-4o-mini",
        context : Optional[CompletionsContext] = None,
        mode : Optional[CompletionsInstructorMode] = None,
        response_model : Optional[CompletionsResponseModel] = None,
        response_format : Optional[CompletionsResponseModel] = None,
        tools : Optional[List[CompletionsToolType]] = None,
        run_tools : Optional[bool] = None,
        tool_choice : Optional[CompletionsToolChoice] = None,
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
    ) -> CompletionsResponse:
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
            raise LLXMException(f"Error building model context: Unable to serialize unknown type: {e}")
        
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
                    messages = messages, model = model, context = context, mode = mode,
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
                    messages = messages, model = model, context = context, mode = mode,
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
    # generators
    # -------------------------------------------------------------------------

    @classmethod
    async def model_async_generate(
        cls,
        messages: Optional[CompletionsMessageType] = None,
        model: CompletionsChatModel = "gpt-4o-mini",
        process: BaseModelGenerationProcess = "batch",
        context: Optional[CompletionsContext] = None,
        batch_size: Optional[int] = None,
        n: Optional[int] = 1,
        mode: Optional[CompletionsInstructorMode] = None,
        tools: Optional[List[CompletionsToolType]] = None,
        run_tools: Optional[bool] = None,
        tool_choice: Optional[CompletionsToolChoice] = None,
        parallel_tool_calls: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = 0,
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
        stream: Optional[bool] = None,
        loader: Optional[bool] = True,
        verbose: Optional[bool] = None
    ) -> Union[Self, List[Self]]:
        """
        Generates one or more instances of the model using the schema definition.
        """
        model_name = cls.__name__
        model_schema = cls.model_json_schema()

        if messages is None:
            messages = f"Generate {n if n > 1 else 'an'} instance{'s' if n > 1 else ''} of {model_name}"

        schema_context = f"""
        You are an expert at generating instances of {model_name}.
        
        # Instructions:
        - Generate valid instances according to the schema
        - Each field must match its type constraints
        - Ensure all required fields are included
        - Generate realistic and coherent data
        
        # Schema:
        {json.dumps(model_schema, indent=2)}
        """

        if context:
            schema_context = f"{context}\n\n{schema_context}"

        if process == "batch":
            ResponseModel = create_model("ResponseModel", items=(List[cls], ...)) if n > 1 else cls
            
            response = await cls.model_async_completion(
                messages=messages,
                model=model,
                context=schema_context,
                mode=mode,
                response_model=ResponseModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )
            
            if n > 1:
                return [cls(**item.model_dump()) for item in response.items]
            return cls(**response.model_dump())

        results = []
        for i in range(n):
            instance_context = schema_context
            if results:
                instance_context += f"\n\nPrevious generations:\n{json.dumps([r.model_dump() for r in results], indent=2)}"
            
            instance_messages = (
                f"Generate instance {i+1} of {n} for {model_name}. "
                "Make this instance different from previous generations."
            ) if results else messages

            response = await cls.model_async_completion(
                messages=instance_messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=cls,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )
            
            results.append(cls(**response.model_dump()))

        return results[0] if n == 1 else results

    @classmethod
    def model_generate(
        cls,
        messages: Optional[CompletionsMessageType] = None,
        model: CompletionsChatModel = "gpt-4o-mini",
        process: BaseModelGenerationProcess = "batch",
        context: Optional[CompletionsContext] = None,
        batch_size: Optional[int] = None,
        n: Optional[int] = 1,
        mode: Optional[CompletionsInstructorMode] = None,
        tools: Optional[List[CompletionsToolType]] = None,
        run_tools: Optional[bool] = None,
        tool_choice: Optional[CompletionsToolChoice] = None,
        parallel_tool_calls: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = 0,
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
        stream: Optional[bool] = None,
        loader: Optional[bool] = True,
        verbose: Optional[bool] = None
    ) -> Union[Self, List[Self]]:
        """
        Generates one or more instances of the model using the schema definition.
        """
        model_name = cls.__name__
        model_schema = cls.model_json_schema()

        if messages is None:
            messages = f"Generate {n if n > 1 else 'an'} instance{'s' if n > 1 else ''} of {model_name}"

        schema_context = f"""
        You are an expert at generating instances of {model_name}.
        
        # Instructions:
        - Generate valid instances according to the schema
        - Each field must match its type constraints
        - Ensure all required fields are included
        - Generate realistic and coherent data
        
        # Schema:
        {json.dumps(model_schema, indent=2)}
        """

        if context:
            schema_context = f"{context}\n\n{schema_context}"

        if process == "batch":
            ResponseModel = create_model("ResponseModel", items=(List[cls], ...)) if n > 1 else cls
            
            response = cls.model_completion(
                messages=messages,
                model=model,
                context=schema_context,
                mode=mode,
                response_model=ResponseModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )
            
            if n > 1:
                return [cls(**item.model_dump()) for item in response.items]
            return cls(**response.model_dump())

        results = []
        for i in range(n):
            instance_context = schema_context
            if results:
                instance_context += f"\n\nPrevious generations:\n{json.dumps([r.model_dump() for r in results], indent=2)}"
            
            instance_messages = (
                f"Generate instance {i+1} of {n} for {model_name}. "
                "Make this instance different from previous generations."
            ) if results else messages

            response = cls.model_completion(
                messages=instance_messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=cls,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )
            
            results.append(cls(**response.model_dump()))

        return results[0] if n == 1 else results


    def model_patch(
        self,
        messages: Optional[CompletionsMessageType] = None,
        model: CompletionsChatModel = "gpt-4o-mini",
        fields: Optional[List[str]] = None,
        context: Optional[CompletionsContext] = None,
        n: Optional[int] = 1,
        mode: Optional[CompletionsInstructorMode] = None,
        tools: Optional[List[CompletionsToolType]] = None,
        run_tools: Optional[bool] = None,
        tool_choice: Optional[CompletionsToolChoice] = None,
        parallel_tool_calls: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = 0,
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
        stream: Optional[bool] = None,
        loader: Optional[bool] = True,
        verbose: Optional[bool] = None
    ) -> Union[Self, List[Self]]:
        """
        Patches or regenerates this model instance, optionally regenerating specific fields.
        Supports generating multiple variations by specifying n > 1.

        Args:
            n (Optional[int]): Number of variations to generate
            fields (Optional[List[str]]): Specific fields to regenerate
            ...other args: Standard completion arguments
        """
        model_name = self.__class__.__name__
        model_schema = self.model_json_schema()
        current_data = self.model_dump()

        # Determine the message based on the presence of fields
        if fields:
            messages = f"Regenerate {n} variations of the following fields of this {model_name}: {', '.join(fields)}" if n > 1 else f"Regenerate the following fields of this {model_name}: {', '.join(fields)}"
        elif messages is None:
            messages = f"Generate {n} unique variations of this {model_name}" if n > 1 else f"Generate a new version of this {model_name}"

        # Build instance context
        instance_context = f"""
        You are an expert at working with {model_name} instances.
        
        # Instructions:
        - Work with the existing instance
        - Maintain data coherence and relationships
        - Ensure all changes are valid per schema
        - Generate coherent and diverse variations when n > 1
        - Each variation should be meaningfully different while maintaining validity
        
        # Schema:
        {json.dumps(model_schema, indent=2)}
        
        # Current Instance:
        {json.dumps(current_data, indent=2)}
        """

        if verbose:
            console.message("Built instance context")

        if context:
            instance_context = f"{context}\n\n{instance_context}"

        if fields:
            # Create a model class with just the specified fields for patching
            field_types = {
                field: (self.model_fields[field].annotation, ...)
                for field in fields
                if field in self.model_fields
            }
            
            if n > 1:
                # For multiple variations, create a wrapper model that contains a list
                PatchModel = create_model(f"{model_name}PatchList", items=(List[create_model(f"{model_name}Patch", **field_types)], ...))
            else:
                PatchModel = create_model(f"{model_name}Patch", **field_types)

            if verbose:
                console.message("Built patch model")

            response = self.model_completion(
                messages=messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=PatchModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )

            if verbose:
                console.message("Generated patch")

            if n > 1:
                # Handle multiple variations
                results = []
                for variation in response.items:
                    updated_data = current_data.copy()
                    variation_data = variation.model_dump()
                    for field in fields:
                        if field in variation_data:
                            updated_data[field] = variation_data[field]
                    results.append(self.__class__(**updated_data))
                return results
            else:
                # Handle single variation
                updated_data = current_data.copy()
                response_data = response.model_dump()
                for field in fields:
                    if field in response_data:
                        updated_data[field] = response_data[field]
                return self.__class__(**updated_data)

        else:
            # Generate entirely new instance(s) based on current
            if n > 1:
                ResponseModel = create_model(f"{model_name}List", items=(List[self.__class__], ...))
            else:
                ResponseModel = self.__class__

            response = self.model_completion(
                messages=messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=ResponseModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )

            if n > 1:
                return [self.__class__(**item.model_dump()) for item in response.items]
            return self.__class__(**response.model_dump())
        
    
    async def model_async_patch(
        self,
        messages: Optional[CompletionsMessageType] = None,
        model: CompletionsChatModel = "gpt-4o-mini",
        fields: Optional[List[str]] = None,
        context: Optional[CompletionsContext] = None,
        n: Optional[int] = 1,
        mode: Optional[CompletionsInstructorMode] = None,
        tools: Optional[List[CompletionsToolType]] = None,
        run_tools: Optional[bool] = None,
        tool_choice: Optional[CompletionsToolChoice] = None,
        parallel_tool_calls: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = 0,
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
        stream: Optional[bool] = None,
        loader: Optional[bool] = True,
        verbose: Optional[bool] = None
    ) -> Union[Self, List[Self]]:
        """
        Generates a new version of this model instance, optionally regenerating specific fields.
        Supports generating multiple variations by specifying n > 1.

        Args:
            n (Optional[int]): Number of variations to generate
            fields (Optional[List[str]]): Specific fields to regenerate
            ...other args: Standard completion arguments
        """
        model_name = self.__class__.__name__
        model_schema = self.model_json_schema()
        current_data = self.model_dump()

        # Determine the message based on the presence of fields
        if fields:
            messages = f"Regenerate {n} variations of the following fields of this {model_name}: {', '.join(fields)}" if n > 1 else f"Regenerate the following fields of this {model_name}: {', '.join(fields)}"
        elif messages is None:
            messages = f"Generate {n} unique variations of this {model_name}" if n > 1 else f"Generate a new version of this {model_name}"

        # Build instance context
        instance_context = f"""
        You are an expert at working with {model_name} instances.
        
        # Instructions:
        - Work with the existing instance
        - Maintain data coherence and relationships
        - Ensure all changes are valid per schema
        - Generate coherent and diverse variations when n > 1
        - Each variation should be meaningfully different while maintaining validity
        
        # Schema:
        {json.dumps(model_schema, indent=2)}
        
        # Current Instance:
        {json.dumps(current_data, indent=2)}
        """

        if verbose:
            console.message("Built instance context")

        if context:
            instance_context = f"{context}\n\n{instance_context}"

        if fields:
            # Create a model class with just the specified fields for patching
            field_types = {
                field: (self.model_fields[field].annotation, ...)
                for field in fields
                if field in self.model_fields
            }
            
            if n > 1:
                # For multiple variations, create a wrapper model that contains a list
                PatchModel = create_model(f"{model_name}PatchList", items=(List[create_model(f"{model_name}Patch", **field_types)], ...))
            else:
                PatchModel = create_model(f"{model_name}Patch", **field_types)

            if verbose:
                console.message("Built patch model")

            response = await self.model_async_completion(
                messages=messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=PatchModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )

            if verbose:
                console.message("Generated patch")

            if n > 1:
                # Handle multiple variations
                results = []
                for variation in response.items:
                    updated_data = current_data.copy()
                    variation_data = variation.model_dump()
                    for field in fields:
                        if field in variation_data:
                            updated_data[field] = variation_data[field]
                    results.append(self.__class__(**updated_data))
                return results
            else:
                # Handle single variation
                updated_data = current_data.copy()
                response_data = response.model_dump()
                for field in fields:
                    if field in response_data:
                        updated_data[field] = response_data[field]
                return self.__class__(**updated_data)

        else:
            # Generate entirely new instance(s) based on current
            if n > 1:
                ResponseModel = create_model(f"{model_name}List", items=(List[self.__class__], ...))
            else:
                ResponseModel = self.__class__

            response = await self.model_async_completion(
                messages=messages,
                model=model,
                context=instance_context,
                mode=mode,
                response_model=ResponseModel,
                tools=tools,
                run_tools=run_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                modalities=modalities,
                prediction=prediction,
                audio=audio,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                extra_headers=extra_headers,
                functions=functions,
                function_call=function_call,
                api_version=api_version,
                model_list=model_list,
                stream=stream,
                loader=loader,
                verbose=verbose
            )

            if n > 1:
                return [self.__class__(**item.model_dump()) for item in response.items]
            return self.__class__(**response.model_dump())
        

# -------------------------------------------------------------------------
# tests
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # tests
    class Sentiment(BaseModel):
        value: str
        confidence: float

    sentiment = Sentiment.model_generate(process = "sequential", verbose = True)

    print(sentiment)

    print(sentiment.model_patch(fields = ["value"], n = 3, verbose = True))
    

