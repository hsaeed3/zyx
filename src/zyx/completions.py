# xnano.completions._client._function_resource
# function resource (extends off xnano.completions._client.base)
# easy shared client for all xnano llm functions, agents & completions

# BASE CLIENT
from enum import Enum
from typing import Any, List, Literal, Generator, Optional, Union, Mapping, Type, Callable

from .base_client import Client, completion

# agents
from .agents import Agents

# FUNCTIONS (lazy loaded (very speedy indeed))
from .functions import (
    coder, classifier, extractor, function_constructor, generator,
    patcher, planner, prompts, queries, question_answer, selector,
    solver, validator
)

# exceptions
from . import _exceptions as exceptions

# function outputs
from .resources.types import model_outputs as outputs

# base client types
from .resources.types.config import client as clientconfig
from .resources.types import completion_create_params as params

import httpx
from pydantic import BaseModel, Field

# for agents
from contextlib import contextmanager


# -- for .prompter()
PROMPT_TYPES = Literal["costar", "tidd-ec", "instruction", "reasoning"]


Store = Type['Store']


# client resource
class Completions(Client):


    """xnano LLM Completions Client"""


    def __init__(
            self,
            provider : Union[Literal["openai", "litellm"], clientconfig.ClientProvider] = "openai",
            # base client args
            api_key : Optional[str] = None,
            base_url : Optional[str] = None,
            organization : Optional[str] = None,
            project : Optional[str] = None,
            timeout : Optional[Union[float, httpx.Timeout]] = clientconfig.DEFAULT_TIMEOUT,
            max_retries : Optional[int] = clientconfig.DEFAULT_MAX_RETRIES,
            default_headers : Optional[Mapping[str, str]] = None,
            default_query : Optional[Mapping[str, object]] = None,
            # pruned httpx args
            verify_ssl : Optional[bool] = None,
            http_args : Optional[Mapping[str, object]] = None,
            # httpx client // only openai supports this
            http_client : Optional[httpx.Client] = None,
            client : Optional[Client] = None,
            # verbosity
            verbose : Optional[bool] = None,
    ):
        
        # super init client
        # no try block needed; handled in _client
        # 'ClientInitializationError'

        super().__init__(
            provider = provider,
            api_key = api_key,
            base_url = base_url,
            organization = organization,
            project = project,
            timeout = timeout,
            max_retries = max_retries,
            default_headers = default_headers,
            default_query = default_query,
            verify_ssl = verify_ssl,
            http_args = http_args,
            http_client = http_client,
            verbose = verbose,
        ) if client is None else client


    def agents(self) -> Agents:
        """Initializes the zyx agents pipeline
        
        Returns:
            Agents: A context manager for working with agents
            
        Example:
            ```python
            with client.agents() as agents:
                agent = agents.add_agent(role="assistant")
                # Work with agents...
            # Agents are automatically cleaned up after context
            ```
        """
        return Agents(client=self)


    def classify(
            self,
            inputs: Union[str, List[str]],
            labels: List[str],
            classification: Literal["single", "multi"] = "single",
            n: int = 1,
            batch_size: int = 3,
            model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            organization: Optional[str] = None,
            mode: params.InstructorMode = "tool_call",
            temperature: Optional[float] = None,
            provider: Optional[Literal["openai", "litellm"]] = "openai",
            progress_bar: Optional[bool] = True,
            verbose: bool = False,
    ) -> List:
        
        """Classify inputs into labels
        
        Example:
            >>> classify(["I love programming in Python", "I like french fries", "I love programming in Julia"], ["code", "food"], classification = "single", batch_size = 2, verbose = True)
            [
                ClassificationResult(text="I love programming in Python", label="code"),
                ClassificationResult(text="I like french fries", label="food")
            ]

        Parameters:
            inputs: Inputs to classify
            labels: Labels to classify inputs into
            classification: Type of classification to perform
            n: Number of classifications to perform
            batch_size: Batch size for classification
            model: Model to use for classification
            api_key: API key to use for classification
            base_url: Base URL to use for classification
            organization: Organization to use for classification
            mode: Mode to use for classification
            temperature: Temperature to use for classification
            provider: Provider to use for classification
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output

        Returns:
            List of ClassificationResults
        """

        try:
            return classifier.classify(
                inputs = inputs,
                labels = labels,
                classification = classification,
                n = n,
                batch_size = batch_size,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                mode = mode,
                temperature = temperature,
                provider = provider,
                progress_bar = progress_bar,
                verbose = verbose,
                client = self
            )
        except Exception as e:
            raise exceptions.ClassifierError(e) from e


    def code(
        self,
        description: str,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        mode: params.InstructorMode = "tool_call",
        temperature: Optional[float] = None,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        progress_bar: Optional[bool] = True,
        return_code: bool = False,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> Any:
        
        """Generate, execute and return results of python code based on a description
        
        Example:
            >>> code("Generate a list of all the files in the current directory", verbose = True)
            ['file1.txt', 'file2.txt', 'file3.txt']

        Parameters:
            description: Description of the code to generate
            model: Model to use for code generation
            api_key: API key to use for code generation
            base_url: Base URL to use for code generation
            organization: Organization to use for code generation
            mode: Mode to use for code generation
            temperature: Temperature to use for code generation
            provider: Provider to use for code generation
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output
            return_code: Whether to return the code
            max_retries: Maximum retries to use for code generation
        Returns:
            Result of the code execution or code if return_code is True
        """

        try:
            return coder.coder(
                description = description,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                mode = mode,
                temperature = temperature,
                provider = provider,
                progress_bar = progress_bar,
                verbose = verbose,
                return_code = return_code,
                max_retries = max_retries,
                client = self
            )
        except Exception as e:
            raise exceptions.CoderError(e) from e
    

    def extract(
        self,
        target: Type[BaseModel],
        text: Union[str, List[str]],
        provider: Literal["litellm", "openai"] = "openai",
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0,
        mode: params.InstructorMode = "tool_call",
        process: Literal["single", "batch"] = "single",
        batch_size: int = 3,
        verbose: bool = False,
        progress_bar: Optional[bool] = True,
    ) -> Union[BaseModel, List[BaseModel]]:

        """Extract structured information from text

        Example:
            >>> extract(User, "John is 20 years old")
            User(name="John", age=20)

        Parameters:
            target: Target model to extract into
            text: Text to extract from
            provider: Provider to use for extraction
            model: Model to use for extraction
            api_key: API key to use for extraction
            base_url: Base URL to use for extraction
            organization: Organization to use for extraction
            max_tokens: Maximum tokens to use for extraction
            max_completion_tokens: Maximum completion tokens to use for extraction
            max_retries: Maximum retries to use for extraction
            temperature: Temperature to use for extraction
            mode: Mode to use for extraction
            process: Process to use for extraction
            batch_size: Batch size to use for extraction
            verbose: Whether to show verbose output
            progress_bar: Whether to show a progress bar

        Returns:
            Extracted model or list of extracted models
        """

        try:
            return extractor.extract(
                target = target,
                text = text,
                provider = provider,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                max_tokens = max_tokens,
                max_completion_tokens = max_completion_tokens,
                max_retries = max_retries,
                temperature = temperature,
                mode = mode,
                process = process,
                batch_size = batch_size,
                verbose = verbose,
                progress_bar = progress_bar,
                client = self
            )
        except Exception as e:
            raise exceptions.ExtractorError(e) from e


    def function(
        self,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        provider: Literal["litellm", "openai"] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        mode: params.InstructorMode = "tool_call",
        mock: bool = False,
        return_code: bool = False,
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
    ) -> Callable[[Callable], Callable]:
        
        """Create a mock or generated runnable python function using LLMs
        
        Parameters:
            model: Model to use for function generation
            provider: Provider to use for function generation
            api_key: API key to use for function generation
            base_url: Base URL to use for function generation
            organization: Organization to use for function generation
            mode: Mode to use for function generation
            mock: Whether to use the mock mode for function generation
            return_code: Whether to return the code for the function
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output

        Returns:
            Mock or generated function
        """

        try:
            return function_constructor.function(
                model = model,
                provider = provider,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                mode = mode,
                mock = mock,
                return_code = return_code,
                progress_bar = progress_bar,
                verbose = verbose,
                client = self
            )
        except Exception as e:
            raise exceptions.FunctionError(e) from e
    

    def generate(
        self,
        target: Type[BaseModel],
        instructions: Optional[str] = None,
        process: Literal["single", "batch"] = "single",
        n: int = 1,
        batch_size: int = 3,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        mode: params.InstructorMode = "tool_call",
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        verbose: bool = False,
        progress_bar: Optional[bool] = True,    
    ) -> Union[BaseModel, List[BaseModel]]:
        
        """Generate a single or batch of pydantic models based on the provided target schema

        Example:
            >>> generate(User, "Generate a user with a name and age")
            User(name="John", age=20)

        Args:
            target: Target model to generate
            instructions: Instructions for generation
            process: Process to use for generation
            n: Number of generations to perform
            batch_size: Batch size for generation
            model: Model to use for generation
            api_key: API key to use for generation
            base_url: Base URL to use for generation
            organization: Organization to use for generation
            temperature: Temperature to use for generation
            max_retries: Maximum retries to use for generation
            mode: Mode to use for generation
            provider: Provider to use for generation
            verbose: Whether to show verbose output
            progress_bar: Whether to show a progress bar

        Returns:
            Generated model or list of generated models
        """

        try:
            return generator.generate(
                target = target,
                instructions = instructions,
                process = process,
                n = n,
                batch_size = batch_size,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                temperature = temperature,
                max_retries = max_retries,
                mode = mode,
                provider = provider,
                verbose = verbose,
                progress_bar = progress_bar,
                client = self
            )
        except Exception as e:
            raise exceptions.GeneratorError(e) from e
    

    def patch(
        self,
        target: BaseModel,
        fields: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,  # Updated default
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        mode: params.InstructorMode = "markdown_json_mode",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
    ) -> BaseModel:
        
        """Patch a pydantic model with text

        Example:
            >>> patch(User(name="John", age=20), "Change the name to Jane")
            User(name="Jane", age=20)

        Args:
            target: Target model to patch
            messages: Messages to use for patching
            model: Model to use for patching
            api_key: API key to use for patching
            base_url: Base URL to use for patching
            organization: Organization to use for patching
            max_tokens: Maximum tokens to use for patching
            max_retries: Maximum retries to use for patching
            temperature: Temperature to use for patching
            mode: Mode to use for patching
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output

        Returns:
            Patched model
        """

        try:
            return patcher.patch(
                target = target,
                fields = fields,
                instructions = instructions,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                max_tokens = max_tokens,
                max_retries = max_retries,
                temperature = temperature,
                mode = mode,
                progress_bar = progress_bar,
                verbose = verbose,
                client = self
            )
        except Exception as e:
            raise exceptions.PatcherError(e) from e
    

    def planner(
        self,
        input: Union[str, Type[BaseModel]],
        instructions: Optional[str] = None,
        process: Literal["single", "batch"] = "single",
        n: int = 1,
        batch_size: int = 3,
        steps: int = 5,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: Optional[float] = None,
        mode: params.InstructorMode = "tool_call",
        max_retries: int = 3,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        verbose: bool = False,
        progress_bar: Optional[bool] = True,
    ) -> Union[outputs.Plan, List[outputs.Plan], Any, List[Any]]:
        
        """Generate a plan or batch of plans based on the input using the Tree of Thoughts method

        Example:
            >>> planner(User, "Generate a user with a name and age")
            Plan(tasks=[Task(description="Generate a user with a name and age", details="")])

        Parameters:
            input: Input to use for planning
            instructions: Instructions for planning
            process: Process to use for planning
            n: Number of plans to generate
            batch_size: Batch size for planning
            steps: Number of steps to use for planning
            model: Model to use for planning
            api_key: API key to use for planning
            base_url: Base URL to use for planning
            organization: Organization to use for planning
            temperature: Temperature to use for planning
            mode: Mode to use for planning
            max_retries: Maximum retries to use for planning
            provider: Provider to use for planning
            verbose: Whether to show verbose output
            progress_bar: Whether to show a progress bar

        Returns:
            Plan or list of plans
        """

        try:
            return planner.planner(
                input = input,
                instructions = instructions,
                process = process,
                n = n,
                batch_size = batch_size,
                steps = steps,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                temperature = temperature,
                mode = mode,
                max_retries = max_retries,
                provider = provider,
                verbose = verbose,
                progress_bar = progress_bar,
                client = self
            )
        except Exception as e:
            raise exceptions.PlannerError(e) from e


    def prompter(
        self,
        instructions: Union[str, List[str]],
        type: PROMPT_TYPES = "costar",
        optimize: bool = False,
        process: Literal["sequential", "batch"] = "sequential",
        n: int = 1,
        batch_size: int = 3,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: Optional[float] = None,
        mode: params.InstructorMode = "tool_call",
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        response_format: Union[Literal["pydantic"], Literal["dict"], None] = None,
        progress_bar: Optional[bool] = True,
        verbose: bool = False,     
    ) -> Union[str, List[str]]:
        
        """Generate or optimize a prompt based on the provided instructions and type

        Args:
            instructions: Instructions for prompt generation
            type: Type of prompt to generate
            optimize: Whether to optimize the prompt
            process: Process to use for prompt generation
            n: Number of prompts to generate
            batch_size: Batch size for prompt generation
            model: Model to use for prompt generation
            api_key: API key to use for prompt generation
            base_url: Base URL to use for prompt generation
            organization: Organization to use for prompt generation
            temperature: Temperature to use for prompt generation
            mode: Mode to use for prompt generation
            max_retries: Maximum retries to use for prompt generation
            max_tokens: Maximum tokens to use for prompt generation
            provider: Provider to use for prompt generation
            response_format: Response format to use for prompt generation
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output

        Returns:
            Generated or optimized prompt
        """

        try:
            return prompts.prompter(
            instructions = instructions,
            type = type,
            optimize = optimize,
            process = process,
            n = n,
            batch_size = batch_size,
            model = model,
            api_key = api_key,
            base_url = base_url,
            organization = organization,
            temperature = temperature,
            max_completion_tokens = max_completion_tokens,
            mode = mode,
            max_retries = max_retries,
            max_tokens = max_tokens,
            provider = provider,
            response_format = response_format,
            progress_bar = progress_bar,
            verbose = verbose,
                client = self
            )
        except Exception as e:
            raise exceptions.PrompterError(e) from e
    

    def query(
        self,
        prompt: str,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        mode: params.InstructorMode = "markdown_json_mode",
        max_retries: int = 3,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        run_tools: Optional[bool] = True,
        tools: Optional[List[Union[Callable, BaseModel]]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        progress_bar: Optional[bool] = True,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        verbose: Optional[bool] = False,  
    ) -> str:
        
        """Solve a problem using a combination of chain-of-thought, high-level concept, and tree-of-thought approaches.

        Example:
            >>> query("What is the capital of France?")
            "Paris"

        Args:
            prompt: Prompt to use for solving the problem
            model: Model to use for solving the problem
            api_key: API key to use for solving the problem
            base_url: Base URL to use for solving the problem
            temperature: Temperature to use for solving the problem
            mode: Mode to use for solving the problem
            max_retries: Maximum retries to use for solving the problem
            organization: Organization to use for solving the problem
            max_tokens: Maximum tokens to use for solving the problem
            run_tools: Whether to run tools for solving the problem
            tools: Tools to use for solving the problem
            parallel_tool_calls: Whether to use parallel tool calls for solving the problem
            tool_choice: Tool choice for solving the problem
            top_p: Top p for solving the problem
            frequency_penalty: Frequency penalty for solving the problem
            presence_penalty: Presence penalty for solving the problem
            stop: Stop for solving the problem
            stream: Whether to stream the response
            progress_bar: Whether to show a progress bar
            provider: Provider to use for solving the problem
            verbose: Whether to show verbose output

        Returns:
            Response from the model
        """

        try:
            return queries.query(
                prompt = prompt,
                model = model,
                api_key = api_key,
                base_url = base_url,
                temperature = temperature,
                mode = mode,
                max_retries = max_retries,
                organization = organization,
                max_tokens = max_tokens,
                run_tools = run_tools,
                tools = tools,
                parallel_tool_calls = parallel_tool_calls,
                tool_choice = tool_choice,
                top_p = top_p,
                frequency_penalty = frequency_penalty,
                presence_penalty = presence_penalty,
                stop = stop,
                stream = stream,
                progress_bar = progress_bar,
                provider = provider,
                verbose = verbose,
                client = self
            )
        except Exception as e:
            raise exceptions.QueryError(e) from e
    

    def qa(
        self,
        input_text: str,
        num_questions: int = 5,
        chunk_size: Optional[int] = 512,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.7,
        mode: params.InstructorMode = "markdown_json_mode",
        max_retries: int = 3,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
        question_instructions: Optional[str] = None,
        answer_instructions: Optional[str] = None,
    ) -> outputs.Dataset:
        
        """Generate a dataset of questions and answers based on the input text.
        """

        try:
            return question_answer.qa(
                input_text = input_text,
                num_questions = num_questions,
                chunk_size = chunk_size,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                temperature = temperature,
                mode = mode,
                max_retries = max_retries,
                provider = provider,
                progress_bar = progress_bar,
                verbose = verbose,
                question_instructions = question_instructions,
                answer_instructions = answer_instructions,
                client = self
            )
        except Exception as e:
            raise exceptions.QaError(e) from e
        

    def select(
        self,
        text: Union[str, List[str], dict, List[dict], Type[Enum], List[Union[str, Enum]]],
        criteria: str,
        selection_type: Literal["single", "multi"] = "single",
        target_type: Optional[Literal["field", "function", "model"]] = None,
        n: int = 1,
        process: Literal["sequential", "batch"] = "sequential",
        context: Optional[str] = None,
        extract_key: Optional[str] = None,
        min_confidence: float = 0.0,
        batch_size: int = 3,
        judge_results: bool = True,
        provider: Literal["litellm", "openai"] = "openai",
        model: Union[str, params.ChatModel] = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0,
        mode: params.InstructorMode = "tool_call",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
        client: Optional[Client] = None,
    ) -> Union[outputs.SelectionResult, outputs.MultiSelectionResult, List[Union[outputs.SelectionResult, outputs.MultiSelectionResult]]]:
        """Selects specific content from text based on given criteria.

        Example:
            ```python
            import zyx
            
            # Single selection
            result = zyx.select(
                "The quick brown fox jumps over the lazy dog",
                "Select the animal that is doing the action",
                selection_type="single"
            )
            
            # Multi selection
            result = zyx.select(
                "The quick brown fox jumps over the lazy dog",
                "Select all animals mentioned",
                selection_type="multi"
            )
            ```

        Args:
            text (Union[str, List[str]]): The text(s) to select content from
            criteria (str): The selection criteria/instructions
            selection_type (Literal["single", "multi"]): Whether to select single or multiple items
            min_confidence (float): Minimum confidence threshold for selections
            batch_size (int): Number of texts to process at once
            provider (str): The LLM provider to use
            model (str): The model to use for selection
            api_key (Optional[str]): API key for the provider
            base_url (Optional[str]): Base URL for API calls
            organization (Optional[str]): Organization ID if applicable
            max_tokens (Optional[int]): Maximum tokens for completion
            temperature (float): Temperature for generation
            mode (str): Response mode for the model
            progress_bar (bool): Whether to show progress bar
            verbose (bool): Whether to show detailed logs
            client (Optional[Client]): Pre-configured client instance

        Returns:
            Union[SelectionResult, MultiSelectionResult, List[Union[SelectionResult, MultiSelectionResult]]]:
                The selected content with confidence scores
        """

        try:
            return selector.select(
                text = text,
                criteria = criteria,
                selection_type = selection_type,
                min_confidence = min_confidence,
                batch_size = batch_size,
                provider = provider,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                temperature = temperature,
                mode = mode,
                progress_bar = progress_bar,
                verbose = verbose,
                process = process,
                context = context,
                extract_key = extract_key,
                judge_results = judge_results,
                target_type = target_type,  
                n = n,
                client = self
            )
        except Exception as e:
            raise exceptions.SelectorError(e) from e
        
    def solve(
        self,
        problem: str,
        high_level_concept: bool = False,
        tree_of_thought: bool = False,
        max_depth: int = 3,
        branching_factor: int = 3,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        mode: params.InstructorMode = "tool_call",
        process: Literal["single", "batch"] = "single",
        batch_size: int = 3,
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
        provider: Optional[Literal["litellm", "openai"]] = "openai",
    ) -> outputs.TreeOfThoughtResult:
        
        """Solve a problem using a combination of chain-of-thought, high-level concept, and tree-of-thought approaches.

        Example:
            >>> solve("What is the capital of France?")
            TreeOfThoughtResult(final_answer="Paris", reasoning_tree=TreeNode(thought=Thought(content="Paris", score=1.0), children=[]))

        Args:
            problem: Problem to solve
            high_level_concept: Whether to use high-level concept
            tree_of_thought: Whether to use tree-of-thought
            max_depth: Maximum depth for tree-of-thought
            branching_factor: Branching factor for tree-of-thought
            model: Model to use for solving the problem
            api_key: API key to use for solving the problem
            base_url: Base URL to use for solving the problem
            organization: Organization to use for solving the problem
            max_tokens: Maximum tokens to use for solving the problem
            max_retries: Maximum retries to use for solving the problem
            temperature: Temperature to use for solving the problem
            mode: Mode to use for solving the problem
            process: Process to use for solving the problem
            batch_size: Batch size for solving the problem
            progress_bar: Whether to show a progress bar
            verbose: Whether to show verbose output
            provider: Provider to use for solving the problem
        """

        try:
            return solver.solve(
                problem = problem,
                high_level_concept = high_level_concept,
                tree_of_thought = tree_of_thought,
                max_depth = max_depth,
                branching_factor = branching_factor,
                model = model,
                api_key = api_key,
                base_url = base_url,
                organization = organization,
                max_tokens = max_tokens,
                max_retries = max_retries,
                temperature = temperature,
                mode = mode,
                process = process,
                batch_size = batch_size,
                progress_bar = progress_bar,
                verbose = verbose,
                provider = provider,
                client = self
            )  
        except Exception as e:
            raise exceptions.SolverError(e) from e
    

    def validate(
        self,
        prompt: str,
        responses: Optional[Union[List[str], str]] = None,
        process: Literal["accuracy", "validate", "fact_check", "guardrails"] = "accuracy",
        schema: Optional[Union[str, dict]] = None,
        regenerate: bool = False,
        model: Union[str, params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        mode: params.InstructorMode = "tool_call",
        max_retries: int = 3,
        organization: Optional[str] = None,
        provider: Optional[Literal["openai", "litellm"]] = "openai",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
        guardrails: Optional[Union[str, List[str]]] = None, 
    ) -> Union[outputs.JudgmentResult, 
               outputs.ValidationResult, 
               outputs.RegeneratedResponse, 
               outputs.FactCheckResult, 
               outputs.GuardrailsResult]:
         
        """
        Judge responses based on accuracy, validate against a schema, fact-check a single response,
        or check against guardrails, with an option to regenerate an optimized response.

        Example:
            ```python
            >>> judge(
                prompt="Explain the concept of quantum entanglement.",
                responses=[
                    "Quantum entanglement is a phenomenon where two particles become interconnected and their quantum states cannot be described independently.",
                    "Quantum entanglement is when particles are really close to each other and move in the same way."
                ],
                process="accuracy",
                verbose=True
            )

            Accuracy Judgment:
            Explanation: The first response is more accurate as it provides a clear definition of quantum entanglement.
            Verdict: The first response is the most accurate.

            Validation Result:
            Is Valid: True
            Explanation: The response adheres to the provided schema.

            Fact-Check Result:
            Is Accurate: True
            Explanation: The response accurately reflects the fact that quantum entanglement occurs when two particles are separated by a large distance but still instantaneously affect each other's quantum states.
            Confidence: 0.95

            Regenerated Response:
            Response: Quantum entanglement is a phenomenon where two particles become interconnected and their quantum states cannot be described independently.
            ```

        Args:
            prompt (str): The original prompt or question.
            responses (List[str]): List of responses to judge, validate, or fact-check.
            process (Literal["accuracy", "validate", "fact_check", "guardrails"]): The type of verification to perform.
            schema (Optional[Union[str, dict]]): Schema for validation or fact-checking (optional for fact_check).
            regenerate (bool): Whether to regenerate an optimized response.
            model (str): The model to use for judgment.
            api_key (Optional[str]): API key for the LLM service.
            base_url (Optional[str]): Base URL for the LLM service.
            temperature (float): Temperature for response generation.
            mode (InstructorMode): Mode for the instructor.
            max_retries (int): Maximum number of retries for API calls.
            organization (Optional[str]): Organization for the LLM service.
            client (Optional[Literal["openai", "litellm"]]): Client to use for API calls.
            verbose (bool): Whether to log verbose output.
            guardrails (Optional[Union[str, List[str]]]): Guardrails for content moderation.

        Returns:
            Union[JudgmentResult, ValidationResult, RegeneratedResponse, FactCheckResult, GuardrailsResult]: The result of the judgment, validation, fact-check, guardrails check, or regeneration.
        """   

        try:
            return validator.validate(
                prompt = prompt,
                responses = responses,
                process = process,
                schema = schema,
                regenerate = regenerate,
                model = model,
                api_key = api_key,
                base_url = base_url,
                temperature = temperature,
                mode = mode,
                max_retries = max_retries,
                organization = organization,
                provider = provider,
                progress_bar = progress_bar,
                verbose = verbose,
                guardrails = guardrails,
                client = self
            )
        except Exception as e:
            raise exceptions.ValidatorError(e) from e
         



