from __future__ import annotations

from ...zyx import messages as messages_utils

from .lib.console import console
from .lib.exception import ZYXException
from .batch import run_structured_batch_completion
from .types.completions.completion_arguments import CompletionArguments
from .types.completions.instructor_mode import InstructorMode
from .types.completions.message import Message
from .types.completions.response_model import ResponseModel
from .types.completions.tools import ToolType, Tool

import instructor

from typing import List, Literal, Optional, Union
from openai.types.chat.chat_completion import ChatCompletion


class BaseClient:

    """Base Completions Resource"""

    # init
    # no client args in init anymore, litellm.completion
    # is the default client/completion method; no class
    # is instantiated anymore
    def __init__(
            self,
            # verbosity
            verbose : bool = False
    ):
        """
        Initializes the completions resource

        Args:
            verbose (bool): Whether to print verbose messages
        """
        
        from litellm import (
            completion, acompletion, batch_completion,
            batch_completion_models_all_responses,
        )

        # set config
        self.verbose = verbose

        # set methods
        try:
            self.litellm_completion = completion
            self.litellm_acompletion = acompletion
            self.litellm_batch_completion = batch_completion
            self.litellm_batch_completion_models_all_responses = batch_completion_models_all_responses
        except Exception as e:
            raise ZYXException(f"Failed to set litellm completion methods: {e}")

        if self.verbose:
            console.message(
                "✅ [green]Successfully initialized [bold]completions[/bold] resource[/green]"
            )
    

    # instructor patch
    def instructor_patch(self, mode : InstructorMode = "tool_call") -> None:
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
                "✅ [green]Successfully patched synchronous completion methods w/ [bold]instructor[/bold] mode: [bold]{mode}[/bold][/green]"
            )


    def instructor_apatch(self, mode : InstructorMode = "tool_call") -> None:
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
                "✅ [green]Successfully patched asynchronous completion methods w/ [bold]instructor[/bold] mode: [bold]{mode}[/bold][/green]"
            )


    def _handle_completion(
            self,
            args : CompletionArguments
    ) -> ChatCompletion:
        """
        Handles synchronous completion

        Args:
            args (CompletionArguments): The completion arguments

        Returns:
            ChatCompletion: The completion response
        """
        
        # format messages
        args.messages = messages_utils.format_messages(args.messages)

        if self.verbose:
            console.message(
                f"💬 [bold]Formatted {len(args.messages)} messages[/bold]"
            )

        # handles structured output
        if self._structured_output:

            # patch instructor in case mode has changed
            try:
                # apply instructor patch
                self.instructor_patch(args.mode)
            except Exception as e:
                raise ZYXException(f"Failed to patch instructor: {e}")

            if not args.response_model and args.response_format:
                args.response_model = args.response_format

        # handles structured output
        if self._structured_output:
            if args.stream:
                return self.instructor_completion.chat.completions.create_partial(
                    messages = args.messages,
                    model = args.model,
                    response_model = args.response_model
                )
            else:
                return self.instructor_completion.chat.completions.create(
                    messages = args.messages,
                    model = args.model,
                    response_model = args.response_model
                )

        # handles unstructured output
        else:
            if self._batch:
                # run litellm batch completion
                return self.litellm_batch_completion(
                    messages = args.messages,
                    model = args.model,
                )
            else:
                # run litellm completion
                return self.litellm_completion(
                    messages = args.messages,
                    model = args.model,
                )
        

    # main -- synchronous completion
    def chat_completion(
            self,
            # messages
            messages : Union[str, Message, List[Message], List[List[Message]]],
            model : str = "gpt-4o-mini",

            mode : InstructorMode = "tool_call",
            response_model : Optional[ResponseModel] = None,
            response_format : Optional[ResponseModel] = None,

            # tool calling
            run_tools : Optional[bool] = None,
            tools : List[ToolType] = None,
            tool_choice : Optional[Literal["auto", "required", "none"]] = None,
            parallel_tool_calls : Optional[bool] = None,

            stream : bool = False,

            **kwargs
    ) -> ChatCompletion:
        """
        Handles completion 

        Args:
            messages (Union[str, Message, List[Message], List[List[Message]]]): The messages to complete
            model (str): The model to use for completion
            response_model (Optional[ResponseModel]): The response model to use for structured output
            response_format (Optional[ResponseModel]): The response format to use for structured output

        Returns:
            ChatCompletion: The completion response
        """

        # internal flags
        self._structured_output = False
        self._batch = False
        
        # determine batch completion
        if isinstance(messages, list) and isinstance(messages[0], list):
            self._batch = True

        # determine structured output
        if response_model or response_format:
            self._structured_output = True

        # set completion arguments
        args = CompletionArguments(
            messages = messages,
            model = model,
            mode = mode,
            response_model = response_model,
            response_format = response_format,
        )

        if isinstance(messages, list) and isinstance(messages[0], list):

            if self._structured_output:
                return run_structured_batch_completion(self, args)
            
        else:

            return self._handle_completion(args)
