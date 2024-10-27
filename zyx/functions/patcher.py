# patch.py

from typing import Optional, List, Union
from pydantic import create_model

from ..base_client import Client
from ..basemodel import BaseModel
from ..resources.types import completion_create_params as params

def patch(
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
    client: Optional[Client] = None,
    verbose: bool = False,
) -> BaseModel:
    """
    Patches specified fields of an existing instance of the BaseModel.
    
    Args:
        target (BaseModel): The instance to patch
        fields (Optional[List[str]]): The fields to patch. If None, patch all fields
        instructions (Optional[str]): Additional instructions for patching
        model (Union[str, ChatModel]): The model to use for patching
        api_key (Optional[str]): The API key to use
        base_url (Optional[str]): The base URL to use  
        organization (Optional[str]): The organization to use
        max_tokens (Optional[int]): Maximum tokens to use
        max_retries (Optional[int]): Maximum retries
        temperature (Optional[float]): Temperature for completions
        mode (InstructorMode): Mode for completions
        progress_bar (Optional[bool]): Whether to show progress bar
        client (Optional[Client]): Optional existing client to use
        verbose (bool): Whether to show verbose output

    Returns:
        BaseModel: The patched instance
    """
    current_data = target.model_dump()
    fields_to_update = fields or list(target.model_fields.keys())

    # Create update model for specified fields
    update_fields = {
        field: (target.model_fields[field].annotation, ...)
        for field in fields_to_update
    }
    BaseModelUpdate = create_model(f"{type(target).__name__}Update", **update_fields)

    completion_client = client or Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider="openai",
        verbose=verbose,
    )

    system_message = (
        f"You are a data patcher. Your task is to update the following fields of an existing {type(target).__name__} instance:\n"
        f"{', '.join(fields_to_update)}\n\n"
        f"Current instance data: {current_data}\n\n"
        f"Model schema: {target.model_json_schema()}\n\n"
        "Provide only the updated values for the specified fields. "
        "Ensure that the updated values comply with the model's schema and constraints."
    )

    user_message = instructions or f"Update the following fields: {', '.join(fields_to_update)}"

    if progress_bar:
        from rich.progress import Progress, SpinnerColumn, TextColumn
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

    # Merge updates into current data
    updated_data = {**current_data, **response.model_dump()}

    # Create and return updated instance
    return type(target)(**updated_data)

# Example usage
if __name__ == "__main__":
    class Person(BaseModel):
        name: str
        age: int
        occupation: str

    person = Person(name="John Doe", age=30, occupation="Engineer")

    updated_person = patch(person, fields=["age", "occupation"], instructions="Update the person's age to 31 and change their occupation to 'Senior Engineer'.", verbose=True)
    print(f"Updated person: {updated_person}")