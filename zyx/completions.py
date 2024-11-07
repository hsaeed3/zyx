import instructor

# interal imports
from .lib.console import console
from .lib.exception import ZYXException
from . import messages as message_utils
from . import response_model as response_model_utils, tool_calling, utils

from .types.completions.completion_arguments import CompletionArguments
from .types.completions.completion_chat_model_param import CompletionChatModelParam
from .types.completions.completion_context import CompletionContext
from .types.completions.completion_instructor_mode import CompletionInstructorMode
from .types.completions.completion_message_param import CompletionMessageParam
from .types.completions.completion_response import CompletionResponse
from .types.completions.completion_response_model_param import CompletionResponseModelParam
from .types.completions.completion_tool import CompletionTool
from .types.completions.completion_tool_choice_param import CompletionToolChoiceParam
from .types.completions.completion_tool_type import CompletionToolType

from typing import List, Optional


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

        # sync
        self.instructor_completion = instructor.from_litellm(self.litellm_completion, mode = instructor.Mode(mode))

        if self.verbose:
            console.message(
                f"✅ [green]Successfully patched synchronous completion methods w/ [bold]instructor[/bold] mode: [bold white]{mode}[/bold white][/green]"
            )


    def instructor_apatch(self, mode : CompletionInstructorMode = "tool_call") -> None:
        """
        Patches the async completion methods with instructor mode

        Args:
            mode (InstructorMode): The mode to patch the completion methods with

        Returns:
            None
        """

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
    ):
        """
        Runs a completion
        """

        if args.response_model is not None:
            try:
                self.instructor_patch(mode = args.mode)

                if self.verbose:
                    console.message(
                        f"✅ [green]Successfully patched instructor w/ mode: [bold white]{args.mode}[/bold white][/green]"
                    )

            except Exception as e:
                raise ZYXException(f"Failed to patch instructor: {e}")
            
            try:
                return self.instructor_completion(
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
            stream : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Runs a completion
        """

        # setup response
        response = None

        # flags
        is_batch_completion = False
        is_tool_execution = False

        # set flags
        if isinstance(messages, list) and isinstance(messages[0], list):
            is_batch_completion = True
        if tools and run_tools is True:
            is_tool_execution = True

        # set response_format as response_model if given instead of response_model
        if response_format:
            console.warning(
                "'response_format' is an allowed argument, but it is preferred to use 'response_model' instead"
            )
            response_model = response_format

        # handle response model
        response_model = response_model_utils.handle_response_model(response_model)

        try:
            # format messages
            messages = message_utils.format_messages(messages)

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
                response = self._run_completion(args)
            except Exception as e:
                raise ZYXException(f"Failed to run completion: {e}")
            
        # return
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
            stream : Optional[bool] = None,
    ) -> CompletionResponse:
        """
        Synchronous completion method
        """

        # run completion
        return self.run_completion(
            messages = messages, model = model, context = context,
            mode = mode, response_model = response_model, response_format = response_format,
            tools = tools, run_tools = run_tools, tool_choice = tool_choice, parallel_tool_calls = parallel_tool_calls,
            api_key = api_key, base_url = base_url, organization = organization, n = n, stream = stream,   
        )