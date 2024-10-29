from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, create_model

from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode

logger = get_logger("system_prompt")

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


def system_prompt(
    instructions: Union[str, List[str]],
    type: PROMPT_TYPES = "costar",
    optimize: bool = False,
    process: Literal["sequential", "batch"] = "sequential",
    n: int = 1,
    batch_size: int = 3,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    temperature: Optional[float] = None,
    mode: InstructorMode = "markdown_json_mode",
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    client: Optional[Literal["openai", "litellm"]] = None,
    response_format: Union[Literal["pydantic"], Literal["dict"], None] = None,
    verbose: bool = False,
) -> Union[BaseModel, str, dict, List[Union[BaseModel, str, dict]]]:
    if verbose:
        logger.info(f"Generating system prompt for {type}")
        logger.info(f"Optimize: {optimize}")
        logger.info(f"Process: {process}")
        logger.info(f"Number of prompts: {n}")
        logger.info(f"Batch size: {batch_size}")

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    response_model = get_response_model(type=type)
    system_prompt = get_system_prompt(type=type)

    if isinstance(instructions, str):
        instructions = [instructions]

    if optimize:
        instructions = [
            f"Optimize the following system prompt:\n\n{instr}"
            for instr in instructions
        ]

    results = []

    if process == "sequential":
        for instr in instructions:
            messages = [
                system_prompt,
                {
                    "role": "user",
                    "content": f"Generate a system prompt for the following instructions:\n\nINSTRUCTIONS:\n{instr}",
                },
            ]
            if results:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Previously generated prompts:\n{results[-1]}",
                    }
                )

            result = completion_client.completion(
                messages=messages,
                model=model,
                response_model=response_model,
                mode=mode,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)
    else:  # batch process
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i : i + batch_size]
            batch_message = (
                "Generate system prompts for the following instructions:\n\n"
            )
            for idx, instr in enumerate(batch, 1):
                batch_message += f"{idx}. {instr}\n\n"

            response_model_batch = create_model(
                "ResponseModel", items=(List[response_model], ...)
            )

            result = completion_client.completion(
                messages=[system_prompt, {"role": "user", "content": batch_message}],
                model=model,
                response_model=response_model_batch,
                mode=mode,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            results.extend(result.items)

    if response_format == "pydantic":
        return results if len(results) > 1 else results[0]

    formatted_results = []
    for prompt in results:
        response_string = []
        for field in prompt.model_fields:
            value = getattr(prompt, field, None)
            if isinstance(value, list):
                formatted_value = "\n".join(f"- {item}" for item in value)
            else:
                formatted_value = value
            response_string.append(f"## {field.capitalize()} ##\n{formatted_value}\n\n")

        if response_format == "dict":
            formatted_results.append(
                {"role": "system", "content": "\n".join(response_string)}
            )
        else:
            formatted_results.append("\n".join(response_string))

    return formatted_results if len(formatted_results) > 1 else formatted_results[0]


if __name__ == "__main__":
    # Example usage
    instructions = [
        "Create a system prompt for a chatbot that helps users with programming questions.",
        "Generate a system prompt for an AI assistant that provides travel recommendations.",
    ]

    result = system_prompt(
        instructions=instructions,
        type="costar",
        optimize=False,
        process="sequential",
        n=2,
        batch_size=2,
        verbose=True,
    )

    print("Generated System Prompts:")
    for idx, prompt in enumerate(result, 1):
        print(f"\nPrompt {idx}:")
        print(prompt)
