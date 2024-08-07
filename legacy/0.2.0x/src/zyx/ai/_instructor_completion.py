# zyx ==============================================================================

from typing import Optional, Type, Union, Literal

PresetTools = Literal["web", "calculator", "shell", "python"]
BaseModel = Type["BaseModel"]
ModelResponse = Type["ModelResponse"]
Assistant = Type["Assistant"]

# --- completion ----------------------------------------------------------------


def instructor_completion(
    messages: Union[str, list[str]],
    model: Optional[str] = "openai/gpt-3.5-turbo",
    response_model: Type["BaseModel"] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: Optional[bool] = True,
    debug: Optional[bool] = False,
    **kwargs,
) -> Union["ModelResponse", "BaseModel"]:
    """
    Runs a litellm completion, tied in with the Instructor framework.

    Parameters:
        messages (Union[str, list[str]]) : The message(s) to send to the model
        model (str) : The model to use for completion
        response_model (Type['BaseModel']) : The response model to use
        base_url (Optional[str]) : The base url for the instructor
        temperature (Optional[float]) : The temperature for the completion
        max_tokens (Optional[int]) : The maximum tokens to use for the completion
        max_retries (Optional[int]) : The maximum retries to use for the completion
        strict (bool) : Whether to use strict mode for the completion

    Returns:
        Union['ModelResponse', 'BaseModel'] : The response from the completion
    """

    if not messages:
        raise ValueError("No messages provided")

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if response_model is None:
        from litellm.main import completion as litellm_completion

        response = litellm_completion(
            model=model,
            messages=messages,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
        return _return_response(response)

    if model.startswith("ollama/"):
        return _ollama_instruct(
            model,
            messages,
            response_model,
            temperature=temperature,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=max_retries,
            strict=strict,
            **kwargs,
        )
    else:
        return _litellm_instruct(
            model,
            messages,
            response_model,
            temperature=temperature,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=max_retries,
            strict=strict,
            **kwargs,
        )


# --- util ----------------------------------------------------------------------


def _return_response(response: "ModelResponse" = None) -> "ModelResponse":
    """
    Returns the response from the completion function

    Parameters:
        response : The response from the completion function
    """
    return response


def _ollama_instruct(
    model: str,
    messages: list[dict],
    response_model: Type["BaseModel"],
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: bool = True,
    **kwargs,
):
    """
    Runs the ollama instructor function

    Parameters:
        model (str) : The model to use for completion
        messages (list[dict]) : A list of messages to send to the model
        response_model (Type['BaseModel']) : The response model to use
        temperature (float) : The temperature to use for completion
        max_tokens (int) : The maximum number of tokens to generate
        max_retries (int) : The maximum number of retries to attempt
        strict (bool) : Whether to raise an exception on failure
        **kwargs : Additional arguments to pass to the completion function
    """
    if not base_url:
        base_url = "http://localhost:11434/v1"

    from instructor.client import from_openai as FromOpenAI
    from instructor.mode import Mode
    from openai import OpenAI

    client = FromOpenAI.from_openai(
        OpenAI(
            base_url=base_url,
            api_key="ollama",
        ),
        mode=Mode.JSON,
    )
    resp = client.chat.completions.create(
        model=model.split("/")[1],  # Remove the Ollama prefix
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        strict=strict,
        **kwargs,
    )
    return resp


def _litellm_instruct(
    model: str,
    messages: list[dict],
    response_model: Type["BaseModel"],
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.5,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    strict: bool = True,
    **kwargs,
):
    """
    Runs the litellm instructor function

    Parameters:
        model (str) : The model to use for completion
        messages (list[dict]) : A list of messages to send to the model
        response_model (Type['BaseModel']) : The response model to use
        temperature (float) : The temperature to use for completion
        max_tokens (int) : The maximum number of tokens to generate
        max_retries (int) : The maximum number of retries to attempt
        strict (bool) : Whether to raise an exception on failure
        **kwargs : Additional arguments to pass to the completion function
    """
    from litellm.main import completion as litellm_completion
    from instructor.client import from_litellm

    client = from_litellm(litellm_completion)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens,
        max_retries=max_retries,
        strict=strict,
        **kwargs,
    )
    return response
