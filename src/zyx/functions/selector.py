# xnano.completions.functions.selector
# llm based content selection/extraction

from ..resources.types import completion_create_params as params
from ..base_client import Client
from .. import _rich as utils
from .validator import validate

from pydantic import BaseModel, create_model, Field
from typing import Any, List, Literal, Optional, Union, Type, Callable, get_type_hints
from enum import Enum
from inspect import getdoc, signature
from rich.progress import Progress, SpinnerColumn, TextColumn

# Internal models for LLM response
class _Selection(BaseModel):
    selected: str
    confidence: float

class _MultiSelection(BaseModel):
    selections: List[str]
    confidence: List[float]

# Final response models
class SelectionResult(BaseModel):
    content : Any
    selected: str
    confidence: float

class MultiSelectionResult(BaseModel):
    content : Any
    selections: List[str]
    confidence: List[float]

def _extract_model_info(model: Type[BaseModel]) -> dict:
    """Extract field names, descriptions, and schema from a Pydantic model."""
    schema = model.model_json_schema()
    fields = {
        name: {
            'description': field.description,
            'type': str(field.annotation),
            'required': field.is_required()
        }
        for name, field in model.model_fields.items()
    }
    return {
        'name': model.__name__,
        'doc': model.__doc__,
        'fields': fields
    }

def _extract_function_info(func: Callable) -> dict:
    """Extract parameter info and docstring from a function."""
    sig = signature(func)
    return {
        'name': func.__name__,
        'doc': getdoc(func),
        'params': {
            name: {
                'type': str(param.annotation),
                'default': None if param.default == param.empty else param.default,
                'required': param.default == param.empty
            }
            for name, param in sig.parameters.items()
        }
    }

def select(
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
) -> Union[SelectionResult, MultiSelectionResult, List[Union[SelectionResult, MultiSelectionResult]]]:
    """Selects specific content from text based on given criteria."""
    
    # Initialize context and valid options
    context_info = ""
    valid_options = []
    original_texts = text

    # Handle different input types
    if isinstance(text, type):
        if issubclass(text, BaseModel):
            model_info = _extract_model_info(text)
            context_info = f"Model: {model_info['name']}\nDescription: {model_info['doc']}\nFields:\n"
            for name, info in model_info['fields'].items():
                context_info += f"- {name}: {info['type']}"
                if info.get('description'):
                    context_info += f" - {info['description']}"
                context_info += "\n"
            valid_options = list(model_info['fields'].keys()) if target_type == "field" else [text.__name__]
            text = valid_options

        elif issubclass(text, Enum):
            valid_options = [e.value for e in text]
            text = valid_options

    elif isinstance(text, (list, tuple)):
        if all(isinstance(item, type) and issubclass(item, BaseModel) for item in text):
            context_info = "Available Models:\n"
            for model_cls in text:
                model_info = _extract_model_info(model_cls)
                context_info += f"\n{model_info['name']}:\n{model_info['doc']}\n"
            valid_options = [model_cls.__name__ for model_cls in text]
            text = valid_options

        elif all(callable(item) for item in text):
            context_info = "Available Functions:\n"
            for func in text:
                func_info = _extract_function_info(func)
                context_info += f"\n{func_info['name']}:\n{func_info['doc']}\n"
            valid_options = [func.__name__ for func in text]
            text = valid_options

        else:
            valid_options = [item.value if isinstance(item, Enum) else str(item) for item in text]
            text = valid_options

    elif callable(text):
        func_info = _extract_function_info(text)
        context_info = f"Function: {func_info['name']}\nDescription: {func_info['doc']}\nParameters:\n"
        for name, info in func_info['params'].items():
            context_info += f"- {name}: {info['type']}\n"
        valid_options = list(func_info['params'].keys()) if target_type == "field" else [text.__name__]
        text = valid_options

    # Handle dict/structured inputs
    if isinstance(text, (dict, list)):
        if extract_key:
            text = [item[extract_key] if isinstance(item, dict) else item[extract_key] 
                   for item in (text if isinstance(text, list) else [text])]
        else:
            text = [str(item) for item in (text if isinstance(text, list) else [text])]

    if isinstance(text, str):
        text = [text]

    if verbose:
        utils.logger.info(f"Selecting from {len(text)} text(s) using {selection_type} selection")
        utils.logger.info(f"Using model: {model}")
        utils.logger.info(f"Selection criteria: {criteria}")

    # Ensure model is string
    model_name = model.value if isinstance(model, BaseModel) else model

    system_message = f"""
    You are a precise content selector. Your task is to select content from text based on specific criteria.
    
    Instructions:
    - Only select content that exactly matches the given criteria
    - Provide a confidence score (0.0 to 1.0) for each selection
    - For single selection, choose the best matching content
    - For multi selection, select all relevant content
    - If nothing matches the criteria, return empty selection with 0.0 confidence
    
    {context_info if context_info else ''}
    {f'Additional Context: {context}' if context else ''}
    Selection Criteria: {criteria}
    """

    if client is None:
        completion_client = Client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            provider=provider,
            verbose=verbose,
        )
    else:
        completion_client = client

    results = []

    if progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task_id = progress.add_task("Selecting content...", total=n)

            for _ in range(n):
                user_message = "Select content from the following text(s):\n\n"
                for idx, t in enumerate(text, 1):
                    user_message += f"{idx}. {t}\n\n"

                result = completion_client.completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    model=model_name,
                    response_model=_Selection if selection_type == "single" else _MultiSelection,
                    mode=mode,
                    temperature=temperature,
                    progress_bar=False,
                )
                
                # Convert to final response model with original text
                if selection_type == "single":
                    if result.confidence >= min_confidence:
                        results.append(SelectionResult(
                            content=original_texts,
                            selected=result.selected,
                            confidence=result.confidence
                        ))
                else:
                    valid_selections = [
                        (s, c) for s, c in zip(result.selections, result.confidence)
                        if c >= min_confidence
                    ]
                    if valid_selections:
                        results.append(MultiSelectionResult(
                            content=original_texts,
                            selections=[s for s, _ in valid_selections],
                            confidence=[c for _, c in valid_selections]
                        ))
                
                progress.update(task_id, advance=1)
    else:
        for _ in range(n):
            user_message = "Select content from the following text(s):\n\n"
            for idx, t in enumerate(text, 1):
                user_message += f"{idx}. {t}\n\n"

            result = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model_name,
                response_model=_Selection if selection_type == "single" else _MultiSelection,
                mode=mode,
                temperature=temperature,
                progress_bar=False,
            )
            
            if selection_type == "single":
                if result.confidence >= min_confidence:
                    results.append(SelectionResult(
                        content=original_texts,
                        selected=result.selected,
                        confidence=result.confidence
                    ))
            else:
                valid_selections = [
                    (s, c) for s, c in zip(result.selections, result.confidence)
                    if c >= min_confidence
                ]
                if valid_selections:
                    results.append(MultiSelectionResult(
                        content=original_texts,
                        selections=[s for s, _ in valid_selections],
                        confidence=[c for _, c in valid_selections]
                    ))

    # Handle no valid results
    if not results:
        return SelectionResult(content=original_texts, selected="", confidence=0.0) if selection_type == "single" \
            else MultiSelectionResult(content=original_texts, selections=[], confidence=[])

    # If we have multiple results and judging is enabled, use validator
    if len(results) > 1 and judge_results:
        responses = [
            f"Selection: {r.selected}\nConfidence: {r.confidence}"
            for r in results
        ]
        
        judgment = validate(
            prompt=f"Original criteria: {criteria}\n{context if context else ''}",
            responses=responses,
            process="accuracy",
            verbose=False,
            progress_bar=False
        )
        
        try:
            best_idx = int(''.join(filter(str.isdigit, judgment.verdict))) - 1
            return results[best_idx]
        except (ValueError, IndexError):
            
            if isinstance(results[0], SelectionResult):
                return max(results, key=lambda x: x.confidence)
            else:
                return max(results, key=lambda x: max(x.confidence))
        
    # Return all results or best result
    if isinstance(results[0], SelectionResult):
        best_result = max(results, key=lambda x: x.confidence)
    else:
        best_result = max(results, key=lambda x: max(x.confidence))

    best_result.content = original_texts
    
    return results if n > 1 and not judge_results else best_result


if __name__ == "__main__":
    
    # Select a field from a Pydantic model
    class User(BaseModel):
        """A user in the system."""
        name: str = Field(description="The user's full name")
        age: int = Field(description="The user's age in years")
        email: str = Field(description="The user's email address")

    field_selection = select(
        text=User,
        criteria="Which field would store the user's contact information?",
        target_type="field",
        selection_type="single"
    )

    print(field_selection)

    print(field_selection.selected)  # Should return "email"

    # Choose between multiple models
    class UserCreate(BaseModel):
        """Model for creating a new user."""
        name: str
        email: str

    class UserUpdate(BaseModel):
        """Model for updating an existing user."""
        name: Optional[str]
        email: Optional[str]

    model_selection = select(
        text=[UserCreate, UserUpdate],
        criteria="Which model would be used for modifying an existing user's information?",
        selection_type="single"
    )


    print(model_selection)


    print(model_selection.selected)  # Should return "UserUpdate"

    # Select a parameter from a function
    def process_data(
        input_file: str,
        output_format: Literal["json", "csv"] = "json",
        compress: bool = False
    ):
        """Process data from input file and convert to specified format."""
        pass

    param_selection = select(
        text=process_data,
        criteria="Which parameter controls the compression of the output?",
        target_type="field",
        selection_type="single"
        )
    
    print(param_selection)

    print(param_selection.selected)  # Should return "compress"
