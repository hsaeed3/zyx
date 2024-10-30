# zyx._basemodel
# llm amplified pydantic models


__all__ = [
    "BaseModel", "Field"
]


from ..lib.utils import logger
from ..resources.types import completion_create_params as params
from ..completions.base_client import Client
from ..lib.environment import ZYX_DEFAULT_MODEL

from pydantic import ConfigDict

import enum
import time
import json
import jsonpatch
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload, TypeVar
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from pydantic import BaseModel as PydanticBaseModel, create_model, Field as PydanticField


T = TypeVar("T", bound=PydanticBaseModel)


class LLMFieldAccessor:
    """Accessor class that provides LLM capabilities to field values"""
    def __init__(self, value: Any, field: Any, instance: 'BaseModel', field_name: str):
        self._value = value
        self._field = field
        self._instance = instance
        self._field_name = field_name

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return repr(self._value)

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __eq__(self, other):
        return self._value == other

    def __iter__(self):
        if hasattr(self._value, '__iter__'):
            return iter(self._value)
        else:
            raise TypeError(f"'{type(self._value).__name__}' object is not iterable")

    def __len__(self):
        if hasattr(self._value, '__len__'):
            return len(self._value)
        else:
            raise TypeError(f"'{type(self._value).__name__}' object has no len()")

    def __getitem__(self, index):
        if hasattr(self._value, '__getitem__'):
            return self._value[index]
        else:
            raise TypeError(f"'{type(self._value).__name__}' object is not subscriptable")

    def __mul__(self, other):
        return self._value * other

    def __truediv__(self, other):
        return self._value / other

    def __floordiv__(self, other):
        return self._value // other

    def __mod__(self, other):
        return self._value % other

    def __pow__(self, other):
        return self._value ** other

    def __lt__(self, other):
        return self._value < other

    def __le__(self, other):
        return self._value <= other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other

    def __neg__(self):
        return -self._value

    def __pos__(self):
        return +self._value

    def __abs__(self):
        return abs(self._value)

    def __round__(self, n=None):
        return round(self._value, n)

    def __floor__(self):
        import math
        return math.floor(self._value)

    def __ceil__(self):
        import math
        return math.ceil(self._value)

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __bool__(self):
        return bool(self._value)

    def __contains__(self, item):
        return item in self._value

    def __call__(self, *args, **kwargs):
        if callable(self._value):
            return self._value(*args, **kwargs)
        raise TypeError(f"'{type(self._value).__name__}' object is not callable")

    def __add__(self, other):
        return self._value + other

    def __sub__(self, other):
        return self._value - other

    def __setattr__(self, name, value):
        if name in {'_value', '_field', '_instance', '_field_name'}:
            super().__setattr__(name, value)
        else:
            setattr(self._value, name, value)
            # Update the instance with the new value
            setattr(self._instance, self._field_name, self._value)

    def __setitem__(self, key, value):
        if hasattr(self._value, '__setitem__'):
            self._value[key] = value
            # Update the instance with the modified value
            self._instance.__dict__[self._field_name] = self._value
        else:
            raise TypeError(f"'{type(self._value).__name__}' object does not support item assignment")

    def __delitem__(self, key):
        if hasattr(self._value, '__delitem__'):
            del self._value[key]
            # Update the instance with the modified value
            self._instance.__dict__[self._field_name] = self._value
        else:
            raise TypeError(f"'{type(self._value).__name__}' object does not support item deletion")

    def __radd__(self, other):
        return other + self._value

    def __rsub__(self, other):
        return other - self._value

    def __rmul__(self, other):
        return other * self._value

    def __rtruediv__(self, other):
        return other / self._value

    def __rfloordiv__(self, other):
        return other // self._value

    def __rmod__(self, other):
        return other % self._value

    def __rpow__(self, other):
        return other ** self._value
    
    def __getattribute__(self, name: str) -> Any:
        if name == "_instance":
            raise AttributeError(f"'_instance' is an invalid attribute for {self._field_name}")
        return super().__getattribute__(name)
    

    def completion(
        self,
        messages: Union[str, List[Dict[str, str]]],
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        **kwargs
    ) -> Any:
        """Run completion for this field"""
        completion_client = Client(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url'),
            organization=kwargs.get('organization'),
            provider="openai",
            verbose=kwargs.get('verbose', False)
        )

        field_context = (
            f"\nField context:"
            f"\nName: {self._field_name}"
            f"\nType: {self._field.annotation}"
            f"\nConstraints: {self._field.json_schema_extra}"
            f"\nCurrent value: {self._value}"
        )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        messages[0]["content"] += field_context

        response = completion_client.completion(
            messages=messages,
            model=model,
            response_model=create_model("FieldResponse", value=(self._field.annotation, ...)),
            **kwargs
        )

        # Update instance with new value
        setattr(self._instance, self._field_name, response.value)
        return response.value


    def patch(
        self,
        instructions: Optional[str] = None,
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        **kwargs
    ) -> Any:
        """Update this field's value"""
        completion_client = Client(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url'),
            organization=kwargs.get('organization'),
            provider="openai",
            verbose=kwargs.get('verbose', False)
        )

        system_message = (
            f"You are updating a field value. Current properties:\n\n"
            f"Field name: {self._field_name}\n"
            f"Type: {self._field.annotation}\n"
            f"Constraints: {self._field.json_schema_extra}\n"
            f"Current value: {self._value}\n\n"
            f"Generate an updated value that complies with the type and constraints."
        )

        user_message = instructions or f"Update the value for the {self._field_name} field."

        response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model=model,
            response_model=create_model("FieldResponse", value=(self._field.annotation, ...)),
            **kwargs
        )

        # Update instance with new value
        setattr(self._instance, self._field_name, response.value)
        return response.value

    def generate(
        self,
        instructions: Optional[str] = None,
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        **kwargs
    ) -> Any:
        """Generate a new value for this field"""
        completion_client = Client(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url'),
            organization=kwargs.get('organization'),
            provider="openai",
            verbose=kwargs.get('verbose', False)
        )

        system_message = (
            f"You are a data generator. Generate a valid value for a field with these properties:\n\n"
            f"Field name: {self._field_name}\n"
            f"Type: {self._field.annotation}\n"
            f"Constraints: {self._field.json_schema_extra}\n\n"
            f"Ensure the generated value complies with the type and constraints."
        )

        user_message = instructions or f"Generate a new value for the {self._field_name} field."

        response = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model=model,
            response_model=create_model("FieldResponse", value=(self._field.annotation, ...)),
            **kwargs
        )

        # Update instance with new value
        setattr(self._instance, self._field_name, response.value)
        return response.value


class BaseModel(PydanticBaseModel):

    def __init__(self, **data):
        super().__init__(**data)
        self._wrapped_fields = {}  # Cache for wrapped fields


    def __setattr__(self, name: str, value: Any):
        # Check if it's a model field
        if name in self.__annotations__:
            # Wrap the value if necessary
            field = self.__annotations__[name]
            if isinstance(value, field):
                value = LLMFieldAccessor(value, field, self, name)
            # Update the wrapped fields cache
            self._wrapped_fields[name] = value
        super().__setattr__(name, value)


    def __getattribute__(self, name: str):
        # First get the _wrapped_fields cache to avoid recursion
        try:
            wrapped_fields = super().__getattribute__('_wrapped_fields')
        except AttributeError:
            wrapped_fields = {}

        # If already wrapped, return from cache
        if name in wrapped_fields:
            return wrapped_fields[name]

        # Get the actual value
        value = super().__getattribute__(name)

        # Check if it's a model field (avoiding recursion by directly accessing dict)
        try:
            model_fields = super().__getattribute__('model_fields')
            if name in model_fields:
                # Create wrapper and cache it
                wrapped = LLMFieldAccessor(
                    value=value,
                    field=model_fields[name],
                    instance=self,
                    field_name=name
                )
                wrapped_fields[name] = wrapped
                return wrapped
        except AttributeError:
            pass

        return value

    @overload
    @classmethod
    def model_generate(
        cls: Type[T],
        instructions: Optional[str] = None,
        n: int = 1,
        process: Literal["batch", "sequential"] = "batch",
        client: Optional[Literal["litellm", "openai"]] = None,
        model : Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0,
        mode: params.InstructorMode = "markdown_json_mode",
        verbose: bool = False,
        progress_bar: Optional[bool] = True,
    ) -> Union[T, List[T]]: ...

    @overload
    def model_generate(
        self: T,
        instructions: Optional[str] = None,
        n: int = 1,
        process: Literal["batch", "sequential"] = "batch",
        client: Optional[Literal["litellm", "openai"]] = None,
        model : Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0,
        mode: params.InstructorMode = "markdown_json_mode",
        verbose: bool = False,
        progress_bar: Optional[bool] = True,
    ) -> Union[T, List[T]]: ...

    @classmethod
    def model_generate(
        cls_or_self: Union[Type[T], T],
        instructions: Optional[str] = None,
        n: int = 1,
        process: Literal["batch", "sequential"] = "batch",
        client: Optional[Literal["litellm", "openai"]] = "openai",
        model : Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0,
        mode: params.InstructorMode = "markdown_json_mode",
        verbose: bool = False,
        progress_bar: Optional[bool] = True,
    ) -> Union[T, List[T]]:
        """Generates instance(s) of the given Pydantic model.

        Example:
            ```python
            class PersonModel(PydanticBaseModel):
                secret_identity: str
                name: str
                age: int

            PersonModel.generate(n=2)
            ```

        Args:
            cls_or_self (Union[Type[T], T]): The class or instance of the Pydantic model.
            instructions (Optional[str]): The instructions to use for the generation.
            n (int): The number of instances to generate.
            model (str): The model to use for the generation.
            api_key (Optional[str]): The API key to use for the generation.
            base_url (Optional[str]): The base URL to use for the generation.
            organization (Optional[str]): The organization to use for the generation.
            max_tokens (Optional[int]): The maximum number of tokens to use for the generation.
            max_retries (int): The maximum number of retries to use for the generation.
            temperature (float): The temperature to use for the generation.
            mode (InstructorMode): The mode to use for the generation.
            progress_bar (Optional[bool]): Whether to print a progress bar.
            verbose (bool): Whether to print verbose output.

        Returns:
            Union[T, List[T]]: The generated instance(s).
        """
        cls = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)

        ResponseModel = cls if n == 1 else create_model("ResponseModel", items=(List[cls], ...))

        system_message = (
            f"You are a data generator. Your task is to generate {n} valid instance(s) of the following Pydantic model:\n\n"
            f"{cls.model_json_schema()}\n\n"
            f"Ensure that all generated instances comply with the model's schema and constraints."
        )

        if isinstance(cls_or_self, BaseModel):
            system_message += f"\n\nUse the following instance as a reference or starting point:\n{cls_or_self.model_dump_json()}"

        system_message += "\nALWAYS COMPLY WITH USER INSTRUCTIONS FOR CONTENT TOPICS & GUIDELINES."

        user_message = instructions or f"Generate {n} instance(s) of the given model."

        if verbose:
            logger.info(f"Template: {system_message}")
            logger.info(f"Instructions: {user_message}")

        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider=client,
            verbose=verbose,
        )

        if process == "batch":

            if progress_bar:

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task_id = progress.add_task("Generating Model(s)...", total=None)

                    response = completion_client.completion(
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        model=model,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        temperature=temperature,
                        mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
                        response_model=ResponseModel,
                        progress_bar=False,
                    )

                    progress.update(task_id, completed=1)

            else:
                response = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    temperature=temperature,
                    mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
                    progress_bar=False,
                )

            return response if n == 1 else response.items
        else:  # Sequential generation
            results = []

            if progress_bar:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task_id = progress.add_task("Generating Model(s)...", total=None)

                    for i in range(n):
                        instance: Dict[str, Any] = {}
                        for field_name, field in cls.model_fields.items():
                            field_system_message = """
You are a data generator tasked with creating valid instances of a Pydantic model based on a provided JSON schema. Your goal is to generate data that strictly adheres to the model's structure and constraints.

Here are the key components for your task:

1. Number of instances to generate:
<instance_count>
{{n}}
</instance_count>

2. Pydantic model JSON schema:
<model_schema>
{{target.model_json_schema()}}
</model_schema>

Instructions:
1. Carefully analyze the provided JSON schema to understand the model's structure, field types, and any constraints.
2. Generate the specified number of instances that comply with the schema.
3. Ensure that all generated instances are valid according to the schema's rules and constraints.
4. Present the generated instances as a collection of JSON objects.

Before generating the data, please use <schema_analysis> tags to break down your approach:
1. Identify and list all required fields and their types
2. Note any optional fields
3. List any constraints or special rules for each field
4. Consider and note any potential challenges in data generation
5. Plan out your approach for generating diverse and valid data

Output Format:
Provide your output as a collection of JSON objects, each representing a valid instance of the model. For example:

```json
[
{
    "field1": "value1",
    "field2": 42,
    "field3": {
    "nested_field": "nested_value"
    }
},
{
    "field1": "another_value",
    "field2": 100,
    "field3": {
    "nested_field": "different_nested_value"
    }
}
]
```

Please proceed with your analysis and data generation.
"""
                            field_user_message = f"Generate a value for the '{field_name}' field."

                            if instance:
                                field_user_message += f"\nCurrent partial instance: {instance}"

                            if i > 0:
                                field_user_message += "\n\nPrevious generations for this field:"
                                for j, prev_instance in enumerate(results[-min(3, i):], 1):
                                    field_user_message += f"\n{j}. {getattr(prev_instance, field_name)}"
                                field_user_message += "\n\nPlease generate a different value from these previous ones."

                            field_user_message += f"\n\nUSER INSTRUCTIONS DEFINED BELOW FOR CONTENT & GUIDELINES\n\n<instructions>\n{instructions or 'No additional instructions provided.'}\n</instructions>"

                            field_response = completion_client.completion(
                                messages=[
                                    {"role": "system", "content": field_system_message},
                                    {"role": "user", "content": field_user_message},
                                ],
                                model=model,
                                max_tokens=max_tokens,
                                max_retries=max_retries,
                                temperature=temperature,
                                mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
                                response_model=create_model("FieldResponse", value=(field.annotation, ...)),
                                progress_bar=False,
                            )
                            instance[field_name] = field_response.value

                        results.append(cls(**instance))

                        progress.update(task_id, completed=i + 1)
            else:

                for i in range(n):
                    instance: Dict[str, Any] = {}
                    for field_name, field in cls.model_fields.items():
                        field_system_message = f"""
You are a data generator tasked with creating valid instances of a Pydantic model based on a provided JSON schema. Your goal is to generate data that strictly adheres to the model's structure and constraints.

Here are the key components for your task:

1. Number of instances to generate:
<instance_count>
{{n}}
</instance_count>

2. Pydantic model JSON schema:
<model_schema>
{{target.model_json_schema()}}
</model_schema>

Instructions:
1. Carefully analyze the provided JSON schema to understand the model's structure, field types, and any constraints.
2. Generate the specified number of instances that comply with the schema.
3. Ensure that all generated instances are valid according to the schema's rules and constraints.
4. Present the generated instances as a collection of JSON objects.

Before generating the data, please use <schema_analysis> tags to break down your approach:
1. Identify and list all required fields and their types
2. Note any optional fields
3. List any constraints or special rules for each field
4. Consider and note any potential challenges in data generation
5. Plan out your approach for generating diverse and valid data

Output Format:
Provide your output as a collection of JSON objects, each representing a valid instance of the model. For example:

```json
[
{
    "field1": "value1",
    "field2": 42,
    "field3": {
    "nested_field": "nested_value"
    }
},
{
    "field1": "another_value",
    "field2": 100,
    "field3": {
    "nested_field": "different_nested_value"
    }
}
]
```

Please proceed with your analysis and data generation.
"""
                        field_user_message = f"Generate a value for the '{field_name}' field."

                        if instance:
                            field_user_message += f"\nCurrent partial instance: {instance}"

                        if i > 0:
                            field_user_message += "\n\nPrevious generations for this field:"
                            for j, prev_instance in enumerate(results[-min(3, i):], 1):
                                field_user_message += f"\n{j}. {getattr(prev_instance, field_name)}"
                            field_user_message += "\n\nPlease generate a different value from these previous ones."

                        field_user_message += f"\n\nUSER INSTRUCTIONS DEFINED BELOW FOR CONTENT & GUIDELINES\n\n<instructions>\n{instructions or 'No additional instructions provided.'}\n</instructions>"

                        field_response = completion_client.completion(
                            messages=[
                                {"role": "system", "content": field_system_message},
                                {"role": "user", "content": field_user_message},
                            ],
                            model=model,
                            max_tokens=max_tokens,
                            max_retries=max_retries,
                            temperature=temperature,
                            mode="markdown_json_mode" if model.startswith(("ollama/", "ollama_chat/")) else mode,
                            response_model=create_model("FieldResponse", value=(field.annotation, ...)),
                            progress_bar=False,
                        )
                        instance[field_name] = field_response.value

                    results.append(cls(**instance))

            return results[0] if n == 1 else results


    @classmethod
    def model_select(
        cls: Type[T],
        field_name: str,
        instructions: Optional[str] = None,
        n: int = 1,
        model : Union[str,  params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        mode: params.InstructorMode = "markdown_json_mode",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
    ) -> Union[enum.Enum, List[enum.Enum]]:
        """Selects values for an Enum field.

        Example:
            ```python
            class PersonModel(PydanticBaseModel):
                secret_identity: str
                name: str
                age: int

            PersonModel.select(field_name="secret_identity", instructions="Select a secret identity for the person.")
            ```

        Args:
            cls (Type[T]): The class of the Pydantic model.
            field_name (str): The name of the Enum field to select values for.
            instructions (Optional[str]): The instructions to use for the selection.
            n (int): The number of values to select.
            model (str): The model to use for the selection.
            api_key (Optional[str]): The API key to use for the selection.
            base_url (Optional[str]): The base URL to use for the selection.
            organization (Optional[str]): The organization to use for the selection.
            max_tokens (Optional[int]): The maximum number of tokens to use for the selection.
            max_retries (int): The maximum number of retries to use for the selection.
            temperature (float): The temperature to use for the selection.
            mode (InstructorMode): The mode to use for the selection.
            progress_bar (Optional[bool]): Whether to print a progress bar.
            verbose (bool): Whether to print verbose output.

        Returns:
            Union[enum.Enum, List[enum.Enum]]: The selected values.
        """
        if field_name not in cls.model_fields or not issubclass(cls.model_fields[field_name].annotation, enum.Enum):
            raise ValueError(f"'{field_name}' is not an Enum field in this model.")

        enum_class = cls.model_fields[field_name].annotation
        enum_values = [e.value for e in enum_class]

        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider="openai",
            verbose=verbose,
        )

        if progress_bar:
            start_time = time()

            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                transient=True
            ) as progress:

                system_message = (
                    f"You are an AI assistant helping to select values for an Enum field.\n\n"
                    f"Field name: {field_name}\n"
                    f"Possible values: {enum_values}\n\n"
                    f"Your task is to select {n} appropriate value(s) from the given options and explain your reasoning."
                )

                user_message = (
                    f"Please select {n} value(s) for the '{field_name}' field.\n\n"
                    f"Instructions: {instructions or 'No additional instructions provided.'}"
                )

                ResponseModel = create_model(
                    "ResponseModel",
                    selections=(List[str], ...),
                    explanations=(List[str], ...)
                )

                response = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    temperature=temperature,
                    mode=mode,
                    response_model=ResponseModel,
                )

                results = [enum_class(selection) for selection in response.selections]

                if verbose:
                    for selection, explanation in zip(results, response.explanations):
                        logger.info(f"Selected: {selection}")
                        logger.info(f"Explanation: {explanation}")

                elapsed_time = time() - start_time
                progress.update(
                    SpinnerColumn(),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    transient=True
                )

        else:

            response = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
                mode=mode,
                response_model=ResponseModel,
            )

            results = [enum_class(selection) for selection in response.selections]

            if verbose:
                for selection, explanation in zip(results, response.explanations):
                    logger.info(f"Selected: {selection}")
                    logger.info(f"Explanation: {explanation}")


        return results[0] if n == 1 else results


    @overload
    @classmethod
    def model_completion(
        cls: Type[T],
        messages: Union[str, List[params.Message]],
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        **kwargs
    ) -> Any:
        ...

    @overload
    def model_completion(
        self: T,
        messages: Union[str, List[params.Message]],
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        **kwargs
    ) -> Any:
        ...

    def model_completion(
        cls_or_self: Union[Type[T], T],
        messages: Union[str, List[Dict[str, str]]],
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        response_model: Union[Optional[Type[PydanticBaseModel]], List[Optional[Type[PydanticBaseModel]]]] = None,
        mode: params.InstructorMode = "tool_call",
        max_retries: Optional[int] = None,
        image: Optional[str] = None,
        run_tools: Optional[bool] = True,
        tools: Optional[List[Union[Dict[str, Any], Type[PydanticBaseModel], Callable]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
        progress_bar: Optional[bool] = True,
        **kwargs
    ) -> params.Completion:
        # Determine if called on class or instance
        if isinstance(cls_or_self, BaseModel):
            self = cls_or_self
            cls = type(self)
        else:
            cls = cls_or_self
            self = None

        # Initialize completion client
        completion_client = Client(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url'),
            organization=kwargs.get('organization'),
            provider="openai",
            verbose=kwargs.get('verbose', False),
        )

        # Format messages with proper system context
        model_context = f"Context: {str(cls.__name__)}"

        if self is not None:
            instance_data = self.model_dump_json()
            # Wrap the instance data in triple backticks to format it as code
            model_context += f"\nInstance data:\n```json\n{instance_data}\n```"

        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": model_context},
                {"role": "user", "content": messages}
            ]
        elif isinstance(messages, list) and messages:
            if messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": model_context})
            else:
                # Ensure that 'content' is a string and not None
                content = messages[0].get('content', '')
                messages[0]['content'] = f"{model_context}\n{content}"

        # Prepare completion arguments
        args = params.CompletionArguments(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            mode=mode,
            max_retries=max_retries,
            stop=stop,
            stream=stream,
            response_model=response_model,
            progress_bar=False,
            tools=tools,
            run_tools=run_tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            **kwargs
        )

        # Execute completion
        if progress_bar:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task_id = progress.add_task("Creating chat completion from model...", total=None)
                response = completion_client.completion(
                    **args.model_dump(),
                )
                progress.update(task_id, completed=1)
        else:
            response = completion_client.completion(
                **args.model_dump(),
            )

        return response



    @classmethod
    def model_regenerate(
        cls: Type[T],
        instance: T,
        fields: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        mode: params.InstructorMode = "markdown_json_mode",
        progress_bar: Optional[bool] = True,
        verbose: bool = False,
    ) -> T:
        """
        Patches specified fields of an existing instance of the Pydantic model using JSON patches generated by an LLM.

        Args:
            instance (T): The instance of the Pydantic model to patch.
            fields (Optional[List[str]]): The fields to patch. If None, patch all fields.
            instructions (Optional[str]): Additional instructions for the patching.
            model (Union[str, ChatModel]): The model to use for the patching.
            api_key (Optional[str]): The API key to use for the patching.
            base_url (Optional[str]): The base URL to use for the patching.
            organization (Optional[str]): The organization to use for the patching.
            max_tokens (Optional[int]): The maximum number of tokens to use for the patching.
            max_retries (int): The maximum number of retries to use for the patching.
            temperature (float): The temperature to use for the patching.
            mode (InstructorMode): The mode to use for the patching.
            progress_bar (Optional[bool]): Whether to display a progress bar.
            verbose (bool): Whether to display verbose output.

        Returns:
            T: The patched instance.
        """
        from pydantic import ValidationError
        import warnings



        # Get current data from the instance
        current_data = instance.model_dump()

        # Get the schema of the model
        schema_json = cls.model_json_schema()

        # Determine fields to update
        fields_to_update = fields or list(cls.model_fields.keys())

        # Prepare system message
        system_message = (
            "You are a data assistant tasked with updating an existing data instance using JSON patches.\n\n"
            "The existing data is:\n"
            f"```json\n{json.dumps(current_data, indent=2)}\n```\n\n"
            "The schema of the data is:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```\n\n"
            "Your task is to generate JSON patches to update the existing data based on the following instructions.\n"
            "Return the patches as a JSON array of JSON Patch operations.\n\n"
            "Make sure that the patched data complies with the schema."
        )

        # Prepare user message
        user_message = instructions or "Update the specified fields."

        if fields_to_update != list(cls.model_fields.keys()):
            user_message += f"\nUpdate the following fields: {fields_to_update}"

        # Initialize completion client
        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider="openai",
            verbose=verbose,
        )

        class JsonPatch(PydanticBaseModel):
            op: Literal["add", "remove", "replace"]
            path: str
            value: Any = None

        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Call completion client
        try:
            response = completion_client.completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
                mode=mode,
                response_model=List[JsonPatch],
                stream=False,
                progress_bar=False
            )
        except Exception as e:
            if verbose:
                logger.error(f"Error during completion: {str(e)}")
            raise ValueError(f"Failed to generate patches: {str(e)}")

        # Extract patches
        patches_list = [patch.model_dump() for patch in response]

        # Apply patches
        json_patch = jsonpatch.JsonPatch(patches_list)
        try:
            updated_data = json_patch.apply(current_data)
        except jsonpatch.JsonPatchConflict as e:
            if verbose:
                logger.error(f"Error applying patches: {e}")
            raise ValueError(f"Failed to apply patches: {str(e)}")

        # Create new instance
        try:
            new_instance = cls(**updated_data)

        except ValidationError as e:
            if verbose:
                logger.error(f"Validation error: {e}")
            raise ValueError(f"Failed to create new instance: {str(e)}")

        return new_instance


    @classmethod
    def model_patch(
        cls: Type[T],
        instance: T,
        fields: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        mode: params.InstructorMode = "markdown_json_mode",
        progress_bar: Optional[bool] = True,
        client: Optional[Client] = None,
        verbose: bool = False,
    ) -> T:
        """
        Patches specified fields of an existing instance of the Pydantic model.

        Args:
            instance (T): The instance of the Pydantic model to patch.
            fields (Optional[List[str]]): The fields to patch. If None, patch all fields.
            instructions (Optional[str]): Additional instructions for the patching.
            model (Union[str, ChatModel]): The model to use for the patching.
            api_key (Optional[str]): The API key to use for the patching.
            base_url (Optional[str]): The base URL to use for the patching.
            organization (Optional[str]): The organization to use for the patching.
            max_tokens (Optional[int]): The maximum number of tokens to use for the patching.
            max_retries (int): The maximum number of retries to use for the patching.
            temperature (float): The temperature to use for the patching.
            mode (InstructorMode): The mode to use for the patching.
            progress_bar (Optional[bool]): Whether to display a progress bar.
            verbose (bool): Whether to display verbose output.

        Returns:
            T: The patched instance.
        """
        current_data = instance.model_dump()
        fields_to_update = fields or list(cls.model_fields.keys())

        # Create a new model for updates
        update_fields = {
            field: (cls.model_fields[field].annotation, ...)
            for field in fields_to_update
        }
        BaseModelUpdate = create_model(f"{cls.__name__}Update", **update_fields)

        completion_client = client or Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider="openai",
            verbose=verbose,
        )

        system_message = (
            f"You are a data patcher. Your task is to update the following fields of an existing {cls.__name__} instance:\n"
            f"{', '.join(fields_to_update)}\n\n"
            f"Current instance data: {current_data}\n\n"
            f"Model schema: {cls.model_json_schema()}\n\n"
            "Provide only the updated values for the specified fields. "
            "Ensure that the updated values comply with the model's schema and constraints."
        )

        user_message = instructions or f"Update the following fields: {', '.join(fields_to_update)}"

        if progress_bar:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task_id = progress.add_task("Patching Model...", total=None)

                response = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    temperature=temperature,
                    mode=mode,
                    response_model=BaseModelUpdate,
                    progress_bar=False,
                )

                progress.update(task_id, completed=1)
        else:
            response = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
                mode=mode,
                response_model=BaseModelUpdate,
                progress_bar=False
            )

        # Merge updates into the current data
        updated_data = {**current_data, **response.model_dump()}

        # Create and return the updated instance
        return cls(**updated_data)
    

    @classmethod
    def model_rag(
        cls,
        prompt: str,
        model: Union[str, params.ChatModel] = ZYX_DEFAULT_MODEL,
        client: Literal["openai", "litellm"] = "openai",
        mode: Optional[params.InstructorMode] = "tool_call",
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

        model_client = cls.model_client or Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider="openai",
            verbose=verbose,
        )

        if not cls.messages:
            cls.messages = [
                {
                    "role": "system",
                "content": f"""
You are an advanced document & data understanding assistant designed to comprehend various types of documents and answer questions about them accurately. Your responses should be optimized for multiple language models and integrated into a retrieval-augmented generation system.

Here is the document you need to analyze:

<document_content>
{cls.model_dump_json(indent=2)}
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

        cls.messages.append({"role": "user", "content": prompt})

        response = model_client.completion(
            messages=cls.messages,
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
            cls.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

        return response


Field = PydanticField


if __name__ == "__main__":

    class TestModel(BaseModel):
        name: str
        age: int


    print(
        TestModel.generate(n=2)
    )
