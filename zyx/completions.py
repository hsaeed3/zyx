import instructor

# interal imports
from .lib.console import console
from .lib.exception import ZYXException
from . import messages as message_utils
from . import response_model as response_model_utils, tool_calling, utils

from .types.completions._openai import ChatCompletionModality, ChatCompletionPredictionContentParam, ChatCompletionAudioParam
from .types.completions.completion_arguments import CompletionArguments
from .types.completions.completion_chat_model_param import CompletionChatModelParam
from .types.completions.completion_context import CompletionContext
from .types.completions.completion_instructor_mode import CompletionInstructorMode
from .types.completions.completion_message_param import CompletionMessageParam
from .types.completions.completion_response import CompletionResponse
from .types.completions.completion_response_model_param import CompletionResponseModelParam
from .types.completions.completion_tool_choice_param import CompletionToolChoiceParam
from .types.completions.completion_tool_type import CompletionToolType

import httpx
import json
from typing import List, Optional, Union


# base client
class Completions:


    """Base Completions Resource"""


    ## init
    # no client args aside from verbosity
    # client uses litellm function methods for completions,
    # not instantiated client
    def __init__(
            self,

            verbose : bool = False
    ):
        """
        Initializes the base completions client

        Args:
            verbose (bool): Whether to print verbose output
        """

        self.verbose = verbose

        try:
            self.import_litellm_methods()
        except Exception as e:
            raise ZYXException(f"Failed to initialize litellm methods: {e}")
        
        if self.verbose:
            console.message(
                "✅ [green]Successfully initialized [bold]completions[/bold] resource[/green]"
            )


    # litellm setup method
    def import_litellm_methods(self) -> None:
        """
        Imports the litellm methods
        """
        import litellm
        from litellm import (
            completion, batch_completion, acompletion,
            batch_completion_models_all_responses
        )

        # drop params
        litellm.drop_params = True

        # set methods
        self.litellm_completion = completion
        self.litellm_batch_completion = batch_completion
        self.litellm_acompletion = acompletion
        self.litellm_batch_completion_models_all_responses = batch_completion_models_all_responses

        self.instructor_completion = None
        self.instructor_acompletion = None
    

    # INSTRUCTOR PATCH METHODS
    # PATCHES COMPLETION / ACOMPLETION FOR SYNC/ASYNC INSTRUCTOR CLIENT
    # instructor patch
    def instructor_patch(self, mode : CompletionInstructorMode = "tool_call") -> None:
        """
        Patches the completion methods with instructor mode
        InstructorMode is a literal list of the string values available in instructor.Mode

        Args:
            mode (InstructorMode): The mode to patch the completion methods with

        Returns:
            None
        """

        if mode is None:
            mode = "tool_call"

        # sync
        if not self.instructor_completion:
            self.instructor_completion = instructor.from_litellm(self.litellm_completion, mode = instructor.Mode(mode))

        if self.verbose:
            console.message(
                f"✅ [green]Successfully patched synchronous completion methods w/ [bold]instructor[/bold] mode: [bold white]{mode if mode is not None else 'tool_call'}[/bold white][/green]"
            )


    def instructor_apatch(self, mode : CompletionInstructorMode = "tool_call") -> None:
        """
        Patches the async completion methods with instructor mode

        Args:
            mode (InstructorMode): The mode to patch the completion methods with

        Returns:
            None
        """

        if not mode:
            mode = "tool_call"

        if not self.instructor_acompletion:
            self.instructor_acompletion = instructor.from_litellm(self.litellm_acompletion, mode = instructor.Mode(mode))

        if self.verbose:
            console.message(
                f"✅ [green]Successfully patched asynchronous completion methods w/ [bold]instructor[/bold] mode: [bold white]{mode}[/bold white][/green]"
            )


    # ------------------------------------------------------------
    # non - batch completion methods
    # ------------------------------------------------------------
    def _run_completion(
            self,
            args : CompletionArguments
    ) -> CompletionResponse:
        """
        Runs a completion
        """

        if args.response_model is not None:
                
            try:
                self.instructor_patch(mode = args.mode)
            except Exception as e:
                raise ZYXException(f"Failed to patch instructor: {e}")
            
            try:
        
                if not args.stream:
                    return self.instructor_completion.chat.completions.create(
                        **utils.build_post_request(args, instructor = True)
                    )
                else:
                    return self.instructor_completion.chat.completions.create_partial(
                        **utils.build_post_request(args, instructor = True)
                    )
                
            except Exception as e:
                raise ZYXException(f"Failed to run instructor completion: {e}")
            
        else:
            try:
                return self.litellm_completion(
                    **utils.build_post_request(args)
                )

            except Exception as e:
                raise ZYXException(f"Failed to run completion: {e}")


    async def _arun_completion(
            self,
            args : CompletionArguments
    ) -> CompletionResponse:
        """
        Runs a completion

        Args:
            args (CompletionArguments): The arguments to run the completion with

        Returns:
            CompletionResponse: The completion response
        """

        if args.response_model is not None:
                
            try:
                self.instructor_apatch(mode = args.mode)
            except Exception as e:
                raise ZYXException(f"Failed to patch instructor: {e}")
            
            try:
        
                if not args.stream:
                    return await self.instructor_acompletion.chat.completions.create(
                        **utils.build_post_request(args, instructor = True)
                    )
                else:
                    return await self.instructor_acompletion.chat.completions.create_partial(
                        **utils.build_post_request(args, instructor = True)
                    )
                
            except Exception as e:
                raise ZYXException(f"Failed to run instructor completion: {e}")
            
        else:
            try:
                return await self.litellm_acompletion(
                    **utils.build_post_request(args)
                )

            except Exception as e:
                raise ZYXException(f"Failed to run completion: {e}")


    # ------------------------------------------------------------
    # batch completion handler
    # ------------------------------------------------------------
    def _run_batch_completion(
            self,
            args : CompletionArguments
    ):
        pass


    # ------------------------------------------------------------
    # base completion handler
    # ------------------------------------------------------------
    def run_completion(
            self,
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Runs a completion

        Returns:
            CompletionResponse: The completion response
        """

        # setup response
        responses = []

        # flags
        ran_tools = False
        is_batch_completion = False
        is_tool_execution = False

        # set flags
        if isinstance(messages, list) and isinstance(messages[0], list):
            is_batch_completion = True
        if tools and run_tools is True:
            is_tool_execution = True

        # ------------------------------------------------------------
        # structured output handling
        # ------------------------------------------------------------

        # set response_format as response_model if given instead of response_model
        if response_format:
            console.warning(
                "'response_format' is an allowed argument, but it is preferred to use 'response_model' instead"
            )
            response_model = response_format

        # handle response model
        if response_model:
            response_model = response_model_utils.handle_response_model(response_model)

        try:
            # format messages
            messages = message_utils.format_messages(messages)

            # add context to messages
            if context:
                messages = message_utils.add_context_to_messages(messages, context)

            if self.verbose:
                if is_batch_completion is False:
                    console.message(
                        f"✅ [green]Successfully formatted {len(messages)} messages[/green]"
                    )
                else:
                    console.message(
                        f"✅ [green]Successfully formatted batch of {len(messages)} messages[/green]"
                    )

        except Exception as e:
            raise ZYXException(f"Failed to validate messages, please ensure they are formatted correctly: {e}")
        
        # build args
        try:
            args = utils.collect_completion_args(locals())
        except Exception as e:
            raise ZYXException(f"Failed to build completion arguments: {e}")
        
        # format tools
        if tools:
            try:

                args.tools = []

                for tool in tools:
                    if isinstance(tool, str):
                        try:
           
                            # build tool
                            args.tools.append(tool_calling.generate_tool(self, tool, model))
           
                        except Exception as e:
                            raise ZYXException(f"Failed to generate tool named: {tool}: {e}")
                    else:
                        args.tools.append(tool_calling.convert_to_tool(tool))
            except Exception as e:
                raise ZYXException(f"Failed to format tools: {e}")
        
        # handle batch
        if is_batch_completion:
            try:

                # TODO: implement
                # tool execution warning for batch completions
                if is_tool_execution:
                    console.warning(
                        "Tool execution is not supported for batch completions yet."
                    )

                # run & return
                return self._run_batch_completion(args)
            
            except Exception as e:
                raise ZYXException(f"Failed to run batch completion: {e}")
            
        # handle non batch
        else:
            try:

                if response_model and tools:
                    args.response_model = None

                response = self._run_completion(args)
            except Exception as e:
                raise ZYXException(f"Failed to run completion: {e}")
            
        # handle tool execution
        if tools and run_tools is True:

            args.messages.append(response.choices[0].message.model_dump())
            responses.append(response)

            # detect tool calls
            if response.choices[0].message.tool_calls:

                # flag
                ran_tools = False

                # iterate through tool calls
                for tool_call in response.choices[0].message.tool_calls:

                    # check if tool name is in args.tools
                    if tool_call.function.name in [tool.name for tool in args.tools]:
                        
                        try:
                            for tool in args.tools:
                                if tool.name == tool_call.function.name and callable(tool.function):
                                    output = tool.function(**json.loads(tool_call.function.arguments))

                                    if output is None:
                                        console.warning(
                                            "Did your function return an output? No output was returned from the tool."
                                        )

                                        output = "Tool executed successfully!"
                                    ran_tools = True

                                    args.messages.append(
                                        {
                                            "role" : "tool",
                                            "content" : json.dumps(f"{str(output)}"),
                                            "tool_call_id" : tool_call.id
                                        }
                                    )

                                    responses.append(
                                        {
                                            "role" : "tool",
                                            "content" : json.dumps(f"{str(output)}"),
                                            "tool_call_id" : tool_call.id
                                        }
                                    )

                                    if self.verbose:
                                        console.message(
                                            f"✅ [green]Successfully ran tool: [bold white]{tool_call.function.name}[/bold white] with args: [bold white]{tool_call.function.arguments}[/bold white][/green]"
                                        )

                        except Exception as e:
                            raise ZYXException(f"Failed to run tool: {e} with args: {tool_call.function.arguments}")

                    # return if none
                    else:
                        console.message(
                            f"No tools called"
                        )

                        return response

            # return if no tools ran
            if not ran_tools:
                console.message(
                        f"No runnable tools called"
                    )

                return response
            
            # get final response
            # add response model if given
            if response_model:
                args.response_model = response_model
                args.tools = None
                args.tool_choice = None
                args.parallel_tool_calls = None

                # NOTE: HACK!
                # instructor does not support tool calls in the final response model
                # so we have to make nice
                args.messages = response_model_utils.make_nice_with_instructor(args.messages)
            # run & return
            try:

                response = self._run_completion(args)

                responses.append(response)
            except Exception as e:
                raise ZYXException(f"Failed to run final completion: {e}")
            
        # return

        if responses:
            return responses
        else:
            return response
    

    # ASYNC
    async def arun_completion(
            self,
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Runs a completion
        """

        # setup response
        responses = []

        # flags
        ran_tools = False
        is_batch_completion = False
        is_tool_execution = False

        # set flags
        if isinstance(messages, list) and isinstance(messages[0], list):
            is_batch_completion = True
        if tools and run_tools is True:
            is_tool_execution = True

        # ------------------------------------------------------------
        # structured output handling
        # ------------------------------------------------------------

        # set response_format as response_model if given instead of response_model
        if response_format:
            console.warning(
                "'response_format' is an allowed argument, but it is preferred to use 'response_model' instead"
            )
            response_model = response_format

        # handle response model
        if response_model:
            response_model = response_model_utils.handle_response_model(response_model)

        try:

            # format messages
            messages = message_utils.format_messages(messages)

            # add context to messages
            if context:
                messages = message_utils.add_context_to_messages(messages, context)

            if self.verbose:
                if is_batch_completion is False:
                    console.message(
                        f"✅ [green]Successfully formatted {len(messages)} messages[/green]"
                    )
                else:
                    console.message(
                        f"✅ [green]Successfully formatted batch of {len(messages)} messages[/green]"
                    )

        except Exception as e:
            raise ZYXException(f"Failed to validate messages, please ensure they are formatted correctly: {e}")
        
        # build args
        try:
            args = utils.collect_completion_args(locals())
        except Exception as e:
            raise ZYXException(f"Failed to build completion arguments: {e}")
        
        # format tools
        if tools:
            try:
                args.tools = [tool_calling.convert_to_tool(tool) for tool in tools]
            except Exception as e:
                raise ZYXException(f"Failed to format tools: {e}")
        
        # handle batch
        if is_batch_completion:
            try:

                # TODO: implement
                # tool execution warning for batch completions
                if is_tool_execution:
                    console.warning(
                        "Tool execution is not supported for batch completions yet."
                    )

                # run & return
                return self._run_batch_completion(args)
            
            except Exception as e:
                raise ZYXException(f"Failed to run batch completion: {e}")
            
        # handle non batch
        else:
            try:

                if response_model and tools:
                    args.response_model = None

                if not run_tools:
                    return await self._arun_completion(args)

                response = self.run_completion(args)
                responses.append(response)
            except Exception as e:
                raise ZYXException(f"Failed to run completion: {e}")
            
        # handle tool execution
        if tools and run_tools is True:

            args.messages.append(response.choices[0].message.model_dump())

            # detect tool calls
            if response.choices[0].message.tool_calls:

                # flag
                ran_tools = False

                # iterate through tool calls
                for tool_call in response.choices[0].message.tool_calls:

                    # check if tool name is in args.tools
                    if tool_call.function.name in [tool.name for tool in args.tools]:
                        
                        try:
                            for tool in args.tools:
                                if tool.name == tool_call.function.name and callable(tool.function):
                                    output = tool.function(**json.loads(tool_call.function.arguments))

                                    if output is None:
                                        console.warning(
                                            "Did your function return an output? No output was returned from the tool."
                                        )

                                        output = "Tool executed successfully!"
                                    ran_tools = True

                                    args.messages.append(
                                        {
                                            "role" : "tool",
                                            "content" : json.dumps(f"{str(output)}"),
                                            "tool_call_id" : tool_call.id
                                        }
                                    )

                                    responses.append(
                                        {
                                            "role" : "tool",
                                            "content" : json.dumps(f"{str(output)}"),
                                            "tool_call_id" : tool_call.id
                                        }
                                    )

                                    if self.verbose:
                                        console.message(
                                            f"✅ [green]Successfully ran tool: [bold white]{tool_call.function.name}[/bold white] with args: [bold white]{tool_call.function.arguments}[/bold white][/green]"
                                        )

                        except Exception as e:
                            raise ZYXException(f"Failed to run tool: {e} with args: {tool_call.function.arguments}")

                    # return if none
                    else:
                        console.message(
                            f"No tools called"
                        )

                        return response

            # return if no tools ran
            if not ran_tools:
                console.message(
                        f"No runnable tools called"
                    )

                return response
            
            # get final response
            # add response model if given
            if response_model:
                args.response_model = response_model
                args.tools = None
                args.tool_choice = None
                args.parallel_tool_calls = None

                # NOTE: HACK!
                # instructor does not support tool calls in the final response model
                # so we have to make nice
                args.messages = response_model_utils.make_nice_with_instructor(args.messages)
            # run & return
            try:

                response = await self._arun_completion(args)
                responses.append(response)
            except Exception as e:
                raise ZYXException(f"Failed to run final completion: {e}")
            
        # return
        if responses:
            return responses
        else:
            return response


    # ------------------------------------------------------------
    # public
    # ------------------------------------------------------------
    def completion(
            self,
            # messages
                # if str, will be formatted as user message
                # if list of list of messages, will be sent as a batch request
            messages : CompletionMessageParam,
            # model -- any litellm model
            model : CompletionChatModelParam = "gpt-4o-mini",
            # context
            context : Optional[CompletionContext] = None,
            # instructor arguments
            # instructor mode patches generation & response parsing
            # method
            # currently instructor does not patch litellm's batch completion so
            # mode will not be used for batch structured completions
            mode : Optional[CompletionInstructorMode] = None,
            # response_model -- default instructor argument
            response_model : Optional[CompletionResponseModelParam] = None,
            # response_format -- converted to response_model if non batch, otherwise used
            response_format : Optional[CompletionResponseModelParam] = None,
            # tool calling
            tools : Optional[List[CompletionToolType]] = None,
            # run tools -- auto executes tool calls
            run_tools : Optional[bool] = None,
            # base tool args
            tool_choice : Optional[CompletionToolChoiceParam] = None,
            parallel_tool_calls : Optional[bool] = None,
            # base completion arguments
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Create a chat completion or completion(s)

        Example:
        ```python
        completion(messages="hi", model="gpt-4o-mini")
        ```

        Args:
            messages (CompletionMessageParam): Messages to send to the model
            model (CompletionChatModelParam): Model to use for the completion
            context (CompletionContext): Context to use for the completion
            mode (CompletionInstructorMode): Instructor mode to use for the completion
            response_model (CompletionResponseModelParam): Response model to use for the completion
            response_format (CompletionResponseModelParam): Response format to use for the completion
            tools (List[CompletionToolType]): Tools to use for the completion
            run_tools (bool): Run tools for the completion
            tool_choice (CompletionToolChoiceParam): Tool choice to use for the completion
            parallel_tool_calls (bool): Parallel tool calls to use for the completion
            api_key (str): API key to use for the completion
            base_url (str): Base URL to use for the completion
            organization (str): Organization to use for the completion
            n (int): Number of completions to use for the completion
            timeout (Union[float, str, httpx.Timeout]): Timeout to use for the completion
            temperature (float): Temperature to use for the completion
            top_p (float): Top P to use for the completion
            stream_options (dict): Stream options to use for the completion
            stop (str): Stop to use for the completion
            max_completion_tokens (int): Max completion tokens to use for the completion
            max_tokens (int): Max tokens to use for the completion
            modalities (List[ChatCompletionModality]): Modalities to use for the completion
            prediction (ChatCompletionPredictionContentParam): Prediction to use for the completion
            audio (ChatCompletionAudioParam): Audio to use for the completion
            presence_penalty (float): Presence penalty to use for the completion
            frequency_penalty (float): Frequency penalty to use for the completion
            logit_bias (dict): Logit bias to use for the completion
            user (str): User to use for the completion
            seed (int): Seed to use for the completion
            logprobs (bool): Logprobs to use for the completion
            top_logprobs (int): Top logprobs to use for the completion
            deployment_id (str): Deployment ID to use for the completion
            extra_headers (dict): Extra headers to use for the completion
            functions (List): Functions to use for the completion
            function_call (str): Function call to use for the completion
            api_version (str): API version to use for the completion
            model_list (list): Model list to use for the completion
            stream (bool): Stream to use for the completion
            verbose (bool): Verbose to use for the completion
        """

        # run completion
        return self.run_completion(
            messages = messages, model = model, context = context,
            mode = mode, response_model = response_model, response_format = response_format,
            tools = tools, run_tools = run_tools, tool_choice = tool_choice, parallel_tool_calls = parallel_tool_calls,
            api_key = api_key, base_url = base_url, organization = organization, n = n, stream = stream,
            timeout = timeout, temperature = temperature, top_p = top_p, stream_options = stream_options, stop = stop,
            max_completion_tokens = max_completion_tokens, max_tokens = max_tokens, modalities = modalities,
            prediction = prediction, audio = audio, presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
            logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs, top_logprobs = top_logprobs,
            deployment_id = deployment_id, extra_headers = extra_headers, functions = functions, function_call = function_call,
            api_version = api_version, model_list = model_list
        )
    
    #static
    @staticmethod
    def _completion(
            # messages
                # if str, will be formatted as user message
                # if list of list of messages, will be sent as a batch request
            messages : CompletionMessageParam,
            # model -- any litellm model
            model : CompletionChatModelParam = "gpt-4o-mini",
            # context
            context : Optional[CompletionContext] = None,
            # instructor arguments
            # instructor mode patches generation & response parsing
            # method
            # currently instructor does not patch litellm's batch completion so
            # mode will not be used for batch structured completions
            mode : Optional[CompletionInstructorMode] = None,
            # response_model -- default instructor argument
            response_model : Optional[CompletionResponseModelParam] = None,
            # response_format -- converted to response_model if non batch, otherwise used
            response_format : Optional[CompletionResponseModelParam] = None,
            # tool calling
            tools : Optional[List[CompletionToolType]] = None,
            # run tools -- auto executes tool calls
            run_tools : Optional[bool] = None,
            # base tool args
            tool_choice : Optional[CompletionToolChoiceParam] = None,
            parallel_tool_calls : Optional[bool] = None,
            # base completion arguments
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
            verbose : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Create a chat completion or completion(s)

        Example:
        ```python
        completion(messages="hi", model="gpt-4o-mini")
        ```

        Args:
            messages (CompletionMessageParam): Messages to send to the model
            model (CompletionChatModelParam): Model to use for the completion
            context (CompletionContext): Context to use for the completion
            mode (CompletionInstructorMode): Instructor mode to use for the completion
            response_model (CompletionResponseModelParam): Response model to use for the completion
            response_format (CompletionResponseModelParam): Response format to use for the completion
            tools (List[CompletionToolType]): Tools to use for the completion
            run_tools (bool): Run tools for the completion
            tool_choice (CompletionToolChoiceParam): Tool choice to use for the completion
            parallel_tool_calls (bool): Parallel tool calls to use for the completion
            api_key (str): API key to use for the completion
            base_url (str): Base URL to use for the completion
            organization (str): Organization to use for the completion
            n (int): Number of completions to use for the completion
            timeout (Union[float, str, httpx.Timeout]): Timeout to use for the completion
            temperature (float): Temperature to use for the completion
            top_p (float): Top P to use for the completion
            stream_options (dict): Stream options to use for the completion
            stop (str): Stop to use for the completion
            max_completion_tokens (int): Max completion tokens to use for the completion
            max_tokens (int): Max tokens to use for the completion
            modalities (List[ChatCompletionModality]): Modalities to use for the completion
            prediction (ChatCompletionPredictionContentParam): Prediction to use for the completion
            audio (ChatCompletionAudioParam): Audio to use for the completion
            presence_penalty (float): Presence penalty to use for the completion
            frequency_penalty (float): Frequency penalty to use for the completion
            logit_bias (dict): Logit bias to use for the completion
            user (str): User to use for the completion
            seed (int): Seed to use for the completion
            logprobs (bool): Logprobs to use for the completion
            top_logprobs (int): Top logprobs to use for the completion
            deployment_id (str): Deployment ID to use for the completion
            extra_headers (dict): Extra headers to use for the completion
            functions (List): Functions to use for the completion
            function_call (str): Function call to use for the completion
            api_version (str): API version to use for the completion
            model_list (list): Model list to use for the completion
            stream (bool): Stream to use for the completion
            verbose (bool): Verbose to use for the completion
        """

        # run completion

        local_args = locals()
        local_args.pop('verbose', None)

        try:
            return Completions(verbose = verbose).run_completion(**local_args)
        except Exception as e:
            raise ZYXException(f"Failed to run completion: {e}")


    # ------------------------------------------------------------
    # async
    # ------------------------------------------------------------
    async def acompletion(
            self,
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
    ):
        """
        Asynchronously create a chat completion or completion(s)

        Example:
        ```python
        completion(messages="hi", model="gpt-4o-mini")
        ```

        Args:
            messages (CompletionMessageParam): Messages to send to the model
            model (CompletionChatModelParam): Model to use for the completion
            context (CompletionContext): Context to use for the completion
            mode (CompletionInstructorMode): Instructor mode to use for the completion
            response_model (CompletionResponseModelParam): Response model to use for the completion
            response_format (CompletionResponseModelParam): Response format to use for the completion
            tools (List[CompletionToolType]): Tools to use for the completion
            run_tools (bool): Run tools for the completion
            tool_choice (CompletionToolChoiceParam): Tool choice to use for the completion
            parallel_tool_calls (bool): Parallel tool calls to use for the completion
            api_key (str): API key to use for the completion
            base_url (str): Base URL to use for the completion
            organization (str): Organization to use for the completion
            n (int): Number of completions to use for the completion
            timeout (Union[float, str, httpx.Timeout]): Timeout to use for the completion
            temperature (float): Temperature to use for the completion
            top_p (float): Top P to use for the completion
            stream_options (dict): Stream options to use for the completion
            stop (str): Stop to use for the completion
            max_completion_tokens (int): Max completion tokens to use for the completion
            max_tokens (int): Max tokens to use for the completion
            modalities (List[ChatCompletionModality]): Modalities to use for the completion
            prediction (ChatCompletionPredictionContentParam): Prediction to use for the completion
            audio (ChatCompletionAudioParam): Audio to use for the completion
            presence_penalty (float): Presence penalty to use for the completion
            frequency_penalty (float): Frequency penalty to use for the completion
            logit_bias (dict): Logit bias to use for the completion
            user (str): User to use for the completion
            seed (int): Seed to use for the completion
            logprobs (bool): Logprobs to use for the completion
            top_logprobs (int): Top logprobs to use for the completion
            deployment_id (str): Deployment ID to use for the completion
            extra_headers (dict): Extra headers to use for the completion
            functions (List): Functions to use for the completion
            function_call (str): Function call to use for the completion
            api_version (str): API version to use for the completion
            model_list (list): Model list to use for the completion
            stream (bool): Stream to use for the completion
        """

        return await self.arun_completion(
            messages = messages, model = model, context = context,
            mode = mode, response_model = response_model, response_format = response_format,
            tools = tools, run_tools = run_tools, tool_choice = tool_choice, parallel_tool_calls = parallel_tool_calls,
            api_key = api_key, base_url = base_url, organization = organization, n = n, stream = stream,
            timeout = timeout, temperature = temperature, top_p = top_p, stream_options = stream_options, stop = stop,
            max_completion_tokens = max_completion_tokens, max_tokens = max_tokens, modalities = modalities,
            prediction = prediction, audio = audio, presence_penalty = presence_penalty, frequency_penalty = frequency_penalty,
            logit_bias = logit_bias, user = user, seed = seed, logprobs = logprobs, top_logprobs = top_logprobs,
            deployment_id = deployment_id, extra_headers = extra_headers, functions = functions, function_call = function_call,
            api_version = api_version, model_list = model_list
        )
    

    # static
    @staticmethod
    def _acompletion(
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
            # openai v1.0+ new params
            seed: Optional[int] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            deployment_id=None,
            extra_headers: Optional[dict] = None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            api_version: Optional[str] = None,
            model_list: Optional[list] = None, 
            stream : Optional[bool] = None,
            verbose : Optional[bool] = None,
    ):
        """
        Asynchronously create a chat completion or completion(s)

        Example:
        ```python
        completion(messages="hi", model="gpt-4o-mini")
        ```

        Args:
            messages (CompletionMessageParam): Messages to send to the model
            model (CompletionChatModelParam): Model to use for the completion
            context (CompletionContext): Context to use for the completion
            mode (CompletionInstructorMode): Instructor mode to use for the completion
            response_model (CompletionResponseModelParam): Response model to use for the completion
            response_format (CompletionResponseModelParam): Response format to use for the completion
            tools (List[CompletionToolType]): Tools to use for the completion
            run_tools (bool): Run tools for the completion
            tool_choice (CompletionToolChoiceParam): Tool choice to use for the completion
            parallel_tool_calls (bool): Parallel tool calls to use for the completion
            api_key (str): API key to use for the completion
            base_url (str): Base URL to use for the completion
            organization (str): Organization to use for the completion
            n (int): Number of completions to use for the completion
            timeout (Union[float, str, httpx.Timeout]): Timeout to use for the completion
            temperature (float): Temperature to use for the completion
            top_p (float): Top P to use for the completion
            stream_options (dict): Stream options to use for the completion
            stop (str): Stop to use for the completion
            max_completion_tokens (int): Max completion tokens to use for the completion
            max_tokens (int): Max tokens to use for the completion
            modalities (List[ChatCompletionModality]): Modalities to use for the completion
            prediction (ChatCompletionPredictionContentParam): Prediction to use for the completion
            audio (ChatCompletionAudioParam): Audio to use for the completion
            presence_penalty (float): Presence penalty to use for the completion
            frequency_penalty (float): Frequency penalty to use for the completion
            logit_bias (dict): Logit bias to use for the completion
            user (str): User to use for the completion
            seed (int): Seed to use for the completion
            logprobs (bool): Logprobs to use for the completion
            top_logprobs (int): Top logprobs to use for the completion
            deployment_id (str): Deployment ID to use for the completion
            extra_headers (dict): Extra headers to use for the completion
            functions (List): Functions to use for the completion
            function_call (str): Function call to use for the completion
            api_version (str): API version to use for the completion
            model_list (list): Model list to use for the completion
            stream (bool): Stream to use for the completion
            verbose (bool): Verbose to use for the completion
        """

        local_args = locals()
        local_args.pop('verbose', None)

        try:
            return Completions(verbose = verbose).acompletion(**local_args)
        except Exception as e:
            raise ZYXException(f"Failed to run completion: {e}")


# functions
completion = Completions._completion
acompletion = Completions._acompletion