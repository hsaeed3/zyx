from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Optional, Type, Literal, TypeVar
from ...client import (
    InstructorMode,
    ToolType,
    Client,
)


T = TypeVar("T", bound=BaseModel)


class Document(BaseModel):
    """
    A document model that can be used to store and query documents.

    Attributes:
        content (Any): The content of the document.
        metadata (Dict[str, Any]): The metadata of the document.
        messages (Optional[List[Dict[str, Any]]]): The messages of the document.
    """

    content: Any
    metadata: Dict[str, Any]
    messages: Optional[List[Dict[str, Any]]] = []

    class Config:
        arbitrary_types_allowed = True

    def setup_messages(self):
        """
        Setup the messages for the document.
        """
        self.messages = [
            {
                "role": "system",
                "content": """
You are a world class document understanding assistant. You are able to 
understand the content of a document and answer questions about it.
""",
            },
            {"role": "user", "content": "What is the document?"},
            {
                "role": "assistant",
                "content": f"""
Here's a full overview of the document! \n
Document Metadata: {self.metadata} \n\n
Document Content: {self.content}
""",
            },
        ]

    def generate(
        self,
        target: Type[T],
        instructions: Optional[str] = None,
        n: int = 1,
        process: Literal["batch", "sequential"] = "batch",
        client: Literal["litellm", "openai"] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0,
        mode: InstructorMode = "markdown_json_mode",
        verbose: bool = False,
    ) -> List[T]:
        """
        Generates a list of instances of the specified Pydantic model using the document's content as context.

        Example:
        ```python
        from zyx import Document

        doc = Document(content="Hello, world!", metadata={"file_name": "file.txt"}, messages=[])
        doc.generate(target=User, instructions="Tell me a joke.")
        ```

        Args:
            target (Type[T]): The Pydantic model to generate instances of.
            instructions (Optional[str]): The instructions for the generation.
            n (int): The number of instances to generate.
            process (Literal["batch", "sequential"]): The process to use for the generation.
            client (Literal["litellm", "openai"]): The client to use for the generation.
            model (str): The model to use for the generation.
            api_key (Optional[str]): The API key to use for the generation.
            base_url (Optional[str]): The base URL to use for the generation.
            organization (Optional[str]): The organization to use for the generation.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            max_retries (int): The maximum number of retries to use for the generation.
            temperature (float): The temperature to use for the generation.
            mode (InstructorMode): The mode to use for the generation.
            verbose (bool): Whether to print the messages to the console.

        Returns:
            List[T]: A list of instances of the specified Pydantic model.
        """
        if not self.messages:
            self.setup_messages()

        if n == 1:
            ResponseModel = target
        else:
            ResponseModel = create_model("ResponseModel", items=(List[target], ...))

        system_message = f"""
        You are a data generator with access to the following document:

        Document Metadata: {self.metadata}
        Document Content: {self.content}

        Your task is to generate {n} valid instance(s) of the following Pydantic model:
        
        {target.model_json_schema()}
        
        Use the document's content as context for generating these instances.
        Ensure that all generated instances comply with the model's schema and constraints.
        """
        user_message = (
            instructions
            if instructions
            else f"Generate {n} instance(s) of the given model using the document's content as context."
        )

        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider=client,
            verbose=verbose,
        )

        if process == "batch":
            response = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
                mode="markdown_json_mode"
                if model.startswith(("ollama/", "ollama_chat/"))
                else mode,
                response_model=ResponseModel,
            )
            return [response] if n == 1 else response.items
        else:  # Sequential generation
            results = []
            for i in range(n):
                instance = {}
                for field_name, field in target.model_fields.items():
                    field_system_message = f"""
                    You are a data generator with access to the following document:

                    Document Metadata: {self.metadata}
                    Document Content: {self.content}

                    Your task is to generate a valid value for the following field:
                    
                    Field name: {field_name}
                    Field type: {field.annotation}
                    Field constraints: {field.json_schema_extra}
                    
                    Use the document's content as context for generating this value.
                    Ensure that the generated value complies with the field's type and constraints.
                    """
                    field_user_message = f"Generate a value for the '{field_name}' field using the document's content as context."
                    if instance:
                        field_user_message += f"\nCurrent partial instance: {instance}"

                    if i > 0:
                        field_user_message += (
                            f"\n\nPrevious generations for this field:"
                        )
                        for j, prev_instance in enumerate(results[-min(3, i) :], 1):
                            field_user_message += (
                                f"\n{j}. {getattr(prev_instance, field_name)}"
                            )
                        field_user_message += "\n\nPlease generate a different value from these previous ones."

                    field_response = completion_client.completion(
                        messages=[
                            {"role": "system", "content": field_system_message},
                            {"role": "user", "content": field_user_message},
                        ],
                        model=model,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        temperature=temperature,
                        mode="markdown_json_mode"
                        if model.startswith(("ollama/", "ollama_chat/"))
                        else mode,
                        response_model=create_model(
                            "FieldResponse", value=(field.annotation, ...)
                        ),
                    )
                    instance[field_name] = field_response.value

                results.append(target(**instance))

            return results

    def completion(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        client: Literal["openai", "litellm"] = "openai",
        response_model: Optional[Type[BaseModel]] = None,
        mode: Optional[InstructorMode] = "tool_call",
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        run_tools: Optional[bool] = True,
        tools: Optional[List[ToolType]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """
        Generates a completion for the document.

        Example:
        ```python
        from zyx import Document

        doc = Document(content="Hello, world!", metadata={"file_name": "file.txt"}, messages=[])
        doc.completion(prompt="Tell me a joke.")
        ```

        Args:
            prompt (str): The prompt to use for the completion.
            model (str): The model to use for the completion.
            client (Literal["openai", "litellm"]): The client to use for the completion.
            response_model (Optional[Type[BaseModel]]): The response model to use for the completion.
            mode (Optional[InstructorMode]): The mode to use for the completion.
            max_retries (Optional[int]): The maximum number of retries to use for the completion.
            api_key (Optional[str]): The API key to use for the completion.
            base_url (Optional[str]): The base URL to use for the completion.
            organization (Optional[str]): The organization to use for the completion.
            run_tools (Optional[bool]): Whether to run the tools for the completion.
            tools (Optional[List[ToolType]]): The tools to use for the completion.
            parallel_tool_calls (Optional[bool]): Whether to run the tools in parallel.
            tool_choice (Optional[Literal["none", "auto", "required"]]): The tool choice to use for the completion.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[float]): The temperature to use for the completion.
            top_p (Optional[float]): The top p to use for the completion.
            frequency_penalty (Optional[float]): The frequency penalty to use for the completion.
            presence_penalty (Optional[float]): The presence penalty to use for the completion.
            stop (Optional[List[str]]): The stop to use for the completion.
            stream (Optional[bool]): Whether to stream the completion.
            verbose (Optional[bool]): Whether to print the messages to the console.

        """
        from ...client import completion

        if not self.messages:
            self.setup_messages()

        self.messages.append({"role": "user", "content": prompt})

        if response_model or tools:
            print("ResponseModel & Tools not supported yet for Document.query()")

        response = completion(
            messages=self.messages,
            model=model,
            client=client,
            mode=mode,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            stop=stop,
            stream=stream,
            verbose=verbose,
        )

        if response:
            self.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

        return response
