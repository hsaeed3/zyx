from typing import Literal, Optional, Union


PROMPT_TYPES = Literal["costar", "tidd-ec"]


def optimize_system_prompt(
    prompt: Union[str, dict[str, str]],
    type: PROMPT_TYPES = "costar",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = None,
    response_format: Union[Literal["pydantic"], Literal["dict"], None] = None,
    verbose: bool = False,
) -> Union[str, dict[str, str]]:
    from .create_system_prompt import create_system_prompt

    prompt = create_system_prompt(
        instructions=f"""
## IMPORTANT INSTRUCTIONS ## \n\n

THE USER HAS REQUESTED YOU OPTIMIZE THEIR EXISTING SYSTEM PROMPT. \n\n

[EXISTING SYSTEM PROMPT]
{prompt}
[/EXISTING SYSTEM PROMPT] \n\n

GENERATE ONLY THE OPTIMIZED SYSTEM PROMPT.
        """,
        type=type,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        response_format=response_format,
        verbose=verbose,
    )

    return prompt
