from typing import Literal, Optional, Union
from pydantic import BaseModel, Field

PROMPT_TYPES = Literal["costar", "tidd-ec"]


class Prompts:
    costar = """
## CONTEXT ##

You are a world class COSTAR prompt creator & engineer.
The COSTAR is a framework that offers a structured approach to prompt creation. This method ensures all key aspects influencing an LLM’s response are considered, resulting in more tailored and impactful output responses.

The COSTAR Framework Explained:
- Context : Providing background information helps the LLM understand the specific scenario.
- Objective (O): Clearly defining the task directs the LLM’s focus.
- Style (S): Specifying the desired writing style aligns the LLM response.
- Tone (T): Setting the tone ensures the response resonates with the required sentiment.
- Audience (A): Identifying the intended audience tailors the LLM’s response to be targeted to an audience.
- Response_Format (R): Providing the response format, like text or json, ensures the LLM outputs, and help build pipelines.

## INSTRUCTIONS ##

- Identify what the user is trying to achieve according to their latest message(s).
- Identify the context of the user's message(s).
- Generate the apprpriate block of text for the COSTAR system prompt, creating a character or persona, that will aid the LLM in achieving the user's objective.

Your response does not always have to include every component of the COSTAR framework, but your prompt should be well-writen, and easy for the LLM to understand.
    """
    tidd_ec = """
## CONTEXT ##

You are a world class Tidd-EC prompt creator & engineer.
The Tidd-EC is a framework that offers a structured approach to prompt creation. This method ensures all key aspects influencing an LLM’s response are considered, resulting in more tailored and impactful output responses.

The Tidd-EC Framework Explained:
- Task : Providing background information helps the LLM understand the specific scenario.
- Instructions : Clearly defining the task directs the LLM’s focus.
- Do : Specifying the desired writing style aligns the LLM response.
- Donts : Setting the tone ensures the response resonates with the required sentiment.

## INSTRUCTIONS ##

- Identify what the user is trying to achieve according to their latest message(s).
- Identify the context of the user's message(s).
- Generate the apprpriate block of text for the COSTAR system prompt, creating a character or persona, that will aid the LLM in achieving the user's objective.

Your response does not always have to include every component of the COSTAR framework, but your prompt should be well-writen, and easy for the LLM to understand.
"""


class CostarSystemPrompt(BaseModel):
    context: Optional[str] = Field(
        ...,
        description="Provide background information to help the LLM understand the specific scenario. If given, must be minimum 4 sentences.",
    )
    objective: Optional[str] = Field(
        ...,
        description="The objective of the user defined task that the LLM is supposed to complete. If given, must be minimum 4 sentences.",
    )
    style: Optional[str] = Field(
        ...,
        description="The user-intended writing style of the response, intended to be produced by the LLM. If given, must be minimum 2 sentences.",
    )
    tone: Optional[str] = Field(
        ...,
        description="The tone of the response, something to help the response resonate with the required sentiment. If given, must be minimum 2 sentences.",
    )
    audience: Optional[str] = Field(
        ...,
        description="The intended audience of the response. If given, must be minimum 2 sentences.",
    )
    response_format: Optional[str] = Field(
        ...,
        description="Examples and desired outcome of the response. If given, must be minimum 2 sentences.",
    )


class TiddECSystemPrompt(BaseModel):
    task: Optional[str] = Field(
        ...,
        description="The description of the character the LLM is supposed to play, as well as the context of the scenario. If given, must be minimum 200 tokens.",
    )
    instructions: Optional[list[str]] = Field(
        ...,
        description="The instructions for the LLM to follow. Must be clear and consise. If given, must be minimum 200 tokens.",
    )
    do: Optional[list[str]] = Field(
        ...,
        description="Specifies actions the LLM should take to successfully complete the prompt. This includes the use of certain language, structures, or information that should be included in the response. Each item must be minimum 20 tokens.",
    )
    donts: Optional[list[str]] = Field(
        ...,
        description="Highlights actions or elements the LLM should avoid in its response. This is essential for preventing common errors or misinterpretations that could lead to inaccurate or irrelevant outputs. Each item must be minimum 20 tokens.",
    )
    examples: Optional[list[str]] = Field(
        ...,
        description="Provides concrete examples of desired outcomes or responses. This component is invaluable for guiding the LLM towards the expected format, style, or content of the response. Each item must be minimum 20 tokens.",
    )


PROMPT_TYPES_MAPPING = {"costar": CostarSystemPrompt, "tidd-ec": TiddECSystemPrompt}


def get_system_prompt(type: PROMPT_TYPES = "costar") -> dict[str, str]:
    prompt_content = getattr(Prompts, type.replace("-", "_"), None)
    if prompt_content is None:
        raise ValueError(f"Invalid prompt type: {type}")
    return {"role": "system", "content": prompt_content}


def get_response_model(type: PROMPT_TYPES = "costar") -> BaseModel:
    return PROMPT_TYPES_MAPPING[type]


def create_system_prompt(
    instructions: str,
    type: PROMPT_TYPES = "costar",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = None,
    response_format: Union[Literal["pydantic"], Literal["dict"], None] = None,
    verbose: bool = False,
) -> Union[BaseModel, str, dict]:
    from ..client import completion

    try:
        if type not in PROMPT_TYPES_MAPPING:
            raise ValueError(
                f"Invalid prompt type: {type}. Must be one of {PROMPT_TYPES_MAPPING.keys()}"
            )

        response_model = get_response_model(type=type)

        if verbose:
            print(
                f"Generating system prompt for {type} with instructions: {instructions}"
            )

        system_prompt = get_system_prompt(type=type)

    except Exception as e:
        raise e

    if verbose:
        print(f"System prompt: {system_prompt}")

    try:
        prompt = completion(
            model=model,
            messages=[
                system_prompt,
                {
                    "role": "user",
                    "content": f"Generate a system prompt for the following instructions:\n\nINSTRUCTIONS:\n{instructions}",
                },
            ],
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            response_model=response_model,
            verbose=verbose,
        )
    except Exception as e:
        raise e

    if prompt is None:
        raise ValueError("Completion function returned None")

    if verbose:
        print(f"Prompt content: {prompt}")

    if response_format == "pydantic":
        return prompt

    response_string = []

    for field in prompt.model_fields:
        value = getattr(prompt, field, None)
        if isinstance(value, list):
            formatted_value = "\n".join(f"- {item}" for item in value)
        else:
            formatted_value = value
        response_string.append(f"## {field.capitalize()} ##\n{formatted_value}\n\n")

    if response_format == "dict":
        return {"role": "system", "content": "\n".join(response_string)}

    return "\n".join(response_string)