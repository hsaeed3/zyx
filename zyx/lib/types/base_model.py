from pydantic import create_model, BaseModel as PydanticBaseModel
from typing import Optional, Literal, List, Type, TypeVar, Union, overload
from ...client import InstructorMode, Client
from ..utils.logger import get_logger


logger = get_logger("base_model")


T = TypeVar("T", bound="BaseModel")


class BaseModel(PydanticBaseModel):
    @overload
    @classmethod
    def generate(
        cls: Type[T],
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
    ) -> List[T]: ...

    @overload
    def generate(
        self: T,
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
    ) -> List[T]: ...

    @classmethod
    def generate(
        cls_or_self: Union[Type[T], T],
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
        Generates a list of instances of the current Pydantic model.

        This method can be used both as a class method and an instance method.

        Example:
            ```python
            class User(BaseModel):
                name: str
                age: int

            # As a class method
            users = User.generate(n=5)

            # As an instance method
            user = User(name="John", age=30)
            more_users = user.generate(n=3)
            ```

        Parameters:
            instructions (Optional[str]): The instructions for the generator.
            n (int): The number of instances to generate.
            process (Literal["batch", "sequential"]): The generation process to use.
            client (Literal["litellm", "openai"]): The client to use for generation.
            model (str): The model to use for generation.
            api_key (Optional[str]): The API key to use for generation.
            base_url (Optional[str]): The base URL to use for generation.
            organization (Optional[str]): The organization to use for generation.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            max_retries (int): The maximum number of retries to make.
            temperature (float): The temperature to use for generation.
            mode (InstructorMode): The mode to use for generation.
            verbose (bool): Whether to print verbose output.

        Returns:
            List[BaseModel]: A list of instances of the current Pydantic model.
        """
        cls = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)

        if n == 1:
            ResponseModel = cls
        else:
            ResponseModel = create_model("ResponseModel", items=(List[cls], ...))

        system_message = f"""
        You are a data generator. Your task is to generate {n} valid instance(s) of the following Pydantic model:

        {cls.model_json_schema()}

        Ensure that all generated instances comply with the model's schema and constraints.
        """

        if isinstance(cls_or_self, BaseModel):
            system_message += f"\n\nUse the following instance as a reference or starting point:\n{cls_or_self.model_dump_json()}"

        system_message += (
            f"\nALWAYS COMPLY WITH USER INSTRUCTIONS FOR CONTENT TOPICS & GUIDELINES."
        )

        user_message = (
            instructions
            if instructions
            else f"Generate {n} instance(s) of the given model."
        )

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
            # Existing logic for batch generation
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
                for field_name, field in cls.model_fields.items():
                    field_system_message = f"""
                    You are a data generator. Your task is to generate a valid value for the following field:

                    Field name: {field_name}
                    Field type: {field.annotation}
                    Field constraints: {field.json_schema_extra}

                    Ensure that the generated value complies with the field's type and constraints.

                    \nALWAYS COMPLY WITH USER INSTRUCTIONS FOR CONTENT TOPICS & GUIDELINES.
                    """
                    field_user_message = (
                        f"Generate a value for the '{field_name}' field."
                    )

                    if instance:
                        field_user_message += f"\nCurrent partial instance: {instance}"

                    # Add information about previous generations
                    if i > 0:
                        field_user_message += (
                            f"\n\nPrevious generations for this field:"
                        )
                        for j, prev_instance in enumerate(results[-min(3, i) :], 1):
                            field_user_message += (
                                f"\n{j}. {getattr(prev_instance, field_name)}"
                            )
                        field_user_message += "\n\nPlease generate a different value from these previous ones."

                    field_user_message += f"""\n\n
                    USER INSTRUCTIONS DEFINED BELOW FOR CONTENT & GUIDELINES

                    <instructions>
                    {instructions if instructions else "No additional instructions provided."}
                    </instructions> \n\n
                    """

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

                results.append(cls(**instance))

            if n == 1:
                return results[0]

            return results


if __name__ == "__main__":

    class TestData(BaseModel):
        compounds: List[str]

    compounds = TestData.generate(
        "make me some data", n=5, process="sequential", verbose=True
    )

    print(compounds)
