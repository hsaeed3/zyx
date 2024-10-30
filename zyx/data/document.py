from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Optional, Type, Literal, TypeVar, Union
from ..resources.types.completions.instructor import InstructorMode
from ..resources.types.completions.arguments import ChatModel
from ..completions import completion, Completions


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
    metadata: Optional[Dict[str, Any]] = {}
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
You are an advanced document & data understanding assistant designed to comprehend various types of documents and answer questions about them accurately. Your responses should be optimized for multiple language models and integrated into a retrieval-augmented generation system.

Here is the document you need to analyze:

<document_metadata>
{{self.metadata}}
</document_metadata>

<document_content>
{{self.content}}
</document_content>

Instructions:
1. Carefully read and analyze both the document metadata and content.
2. When a user asks a question, follow these steps:
   a. Wrap your analysis process inside <analysis> tags.
   b. In your analysis, consider the following:
      - Quote relevant parts of the document metadata and content
      - Identify any potential ambiguities or multiple interpretations of the question
      - Evaluate the most accurate and concise way to answer the question
      - Assess your confidence level in the answer based on the available information
   c. Provide your final answer after the analysis process.

3. Adhere to these guidelines:
   - Be thorough in your analysis and precise in your answers.
   - Stick strictly to the information provided in the document. Do not introduce external knowledge or make assumptions beyond what's given.
   - If the document doesn't contain enough information to answer a question fully, state this clearly.
   - If a question is unclear or could have multiple interpretations, ask for clarification before attempting to answer.

Remember, your role is to assist users in understanding the document by answering their questions accurately and helpfully. Always base your responses on the document's content and metadata.

Please wait for a user question to begin.
""",
            },
        ]

    def generate(
        self,
        target: Type[T],
        instructions: Optional[str] = None,
        n: int = 1,
        process: Literal["batch", "sequential"] = "batch",
        client: Literal["litellm", "openai"] = "openai",
        model: Union[str, ChatModel] = "gpt-4o-mini",
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

        completion_client = Completions(
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
You are an advanced AI assistant specialized in data generation and analysis. Your task is to generate a valid value for a specific field based on the content of a given document. Please follow these instructions carefully:

1. Document Content:
<document_content>
{self.content}
</document_content>

2. Document Metadata:
<document_metadata>
{self.metadata}
</document_metadata>

3. Field Information:
You need to generate a value for the following field:
<field_name>{field_name}</field_name>

The type of this field is:
<field_type>{field.annotation}</field_type>

The constraints for this field are:
<field_constraints>{field.json_schema_extra}</field_constraints>

4. Task Instructions:
a) Carefully analyze the document content and metadata.
b) Consider the field name, type, and constraints.
c) Generate a value that is relevant to the document and complies with the field's type and constraints.
d) Ensure that the generated value is based on the information found in the document.

5. Reasoning Process:
Before providing your final answer, wrap your reasoning process inside <reasoning> tags. Follow these steps:
- Identify and quote relevant information from the document content and metadata.
- Analyze how this information relates to the required field.
- List the key constraints and requirements for the field.
- Explain how you will ensure the generated value is both relevant and compliant.
- If multiple options are possible, consider the pros and cons of each.

6. Output Format:
After your reasoning, provide your final generated value in the following format:

Generated Value: [Your generated value here]

Remember, the generated value should be a single item that fits the field type and constraints.

Now, please proceed with the task. Start by analyzing the document and showing your reasoning process, then provide the generated value.
""" 

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
        model: Union[str, ChatModel] = "gpt-4o-mini",
        client: Literal["openai", "litellm"] = "openai",
        response_model: Optional[Type[BaseModel]] = None,
        mode: Optional[InstructorMode] = "tool_call",
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
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

        if not self.messages:
            self.messages = [
                {
                    "role": "system",
                "content": """
You are an advanced document & data understanding assistant designed to comprehend various types of documents and answer questions about them accurately. Your responses should be optimized for multiple language models and integrated into a retrieval-augmented generation system.

Here is the document you need to analyze:

<document_metadata>
{{self.metadata}}
</document_metadata>

<document_content>
{{self.content}}
</document_content>

Instructions:
1. Carefully read and analyze both the document metadata and content.
2. When a user asks a question, follow these steps:
   a. Wrap your analysis process inside <analysis> tags.
   b. In your analysis, consider the following:
      - Quote relevant parts of the document metadata and content
      - Identify any potential ambiguities or multiple interpretations of the question
      - Evaluate the most accurate and concise way to answer the question
      - Assess your confidence level in the answer based on the available information
   c. Provide your final answer after the analysis process.

3. Adhere to these guidelines:
   - Be thorough in your analysis and precise in your answers.
   - Stick strictly to the information provided in the document. Do not introduce external knowledge or make assumptions beyond what's given.
   - If the document doesn't contain enough information to answer a question fully, state this clearly.
   - If a question is unclear or could have multiple interpretations, ask for clarification before attempting to answer.

Remember, your role is to assist users in understanding the document by answering their questions accurately and helpfully. Always base your responses on the document's content and metadata.

Please wait for a user question to begin.
""",
                },
            ]

        self.messages.append({"role": "user", "content": prompt})

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
            stop=stop,
            stream=stream,
            verbose=verbose,
        )

        if response:
            self.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

        return response