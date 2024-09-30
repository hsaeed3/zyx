from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

from ....lib.utils.logger import get_logger
from ....client import Client, InstructorMode


logger = get_logger("judge")


class JudgmentResult(BaseModel):
    explanation: str
    verdict: str


class ValidationResult(BaseModel):
    is_valid: bool
    explanation: str


class RegeneratedResponse(BaseModel):
    response: str


class FactCheckResult(BaseModel):
    is_accurate: bool
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)


def judge(
    prompt: str,
    responses: Optional[Union[List[str], str]] = None,
    process: Literal["accuracy", "validate", "fact_check"] = "accuracy",
    schema: Optional[Union[str, dict]] = None,
    regenerate: bool = False,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    mode: InstructorMode = "markdown_json_mode",
    max_retries: int = 3,
    organization: Optional[str] = None,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
    guardrails: Optional[Union[str, List[str]]] = None,
) -> Union[JudgmentResult, ValidationResult, RegeneratedResponse, FactCheckResult]:
    """
    Judge responses based on accuracy, validate against a schema, or fact-check a single response,
    with an option to regenerate an optimized response.

    Example:
        ```python
        >>> judge(
            prompt="Explain the concept of quantum entanglement.",
            responses=[
                "Quantum entanglement is a phenomenon where two particles become interconnected and their quantum states cannot be described independently.",
                "Quantum entanglement is when particles are really close to each other and move in the same way."
            ],
            process="accuracy",
            verbose=True
        )

        Accuracy Judgment:
        Explanation: The first response is more accurate as it provides a clear definition of quantum entanglement.
        Verdict: The first response is the most accurate.

        Validation Result:
        Is Valid: True
        Explanation: The response adheres to the provided schema.

        Fact-Check Result:
        Is Accurate: True
        Explanation: The response accurately reflects the fact that quantum entanglement occurs when two particles are separated by a large distance but still instantaneously affect each other's quantum states.
        Confidence: 0.95

        Regenerated Response:
        Response: Quantum entanglement is a phenomenon where two particles become interconnected and their quantum states cannot be described independently.
        ```

    Args:
        prompt (str): The original prompt or question.
        responses (List[str]): List of responses to judge, validate, or fact-check.
        process (Literal["accuracy", "validate", "fact_check"]): The type of verification to perform.
        schema (Optional[Union[str, dict]]): Schema for validation or fact-checking (optional for fact_check).
        regenerate (bool): Whether to regenerate an optimized response.
        model (str): The model to use for judgment.
        api_key (Optional[str]): API key for the LLM service.
        base_url (Optional[str]): Base URL for the LLM service.
        temperature (float): Temperature for response generation.
        mode (InstructorMode): Mode for the instructor.
        max_retries (int): Maximum number of retries for API calls.
        organization (Optional[str]): Organization for the LLM service.
        client (Optional[Literal["openai", "litellm"]]): Client to use for API calls.
        verbose (bool): Whether to log verbose output.
        guardrails (Optional[Union[str, List[str]]]): Guardrails for content moderation.

    Returns:
        Union[JudgmentResult, ValidationResult, RegeneratedResponse, FactCheckResult]: The result of the judgment, validation, fact-check, or regeneration.
    """
    if verbose:
        logger.info(f"Judging responses for prompt: {prompt}")
        logger.info(f"process: {process}")
        logger.info(f"Regenerate: {regenerate}")

    if isinstance(responses, str):
        responses = [responses]

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    if process == "accuracy":
        system_message = (
            "You are an impartial judge evaluating responses to a given prompt. "
            "Compare the responses and determine which one is the most accurate, helpful, and relevant. "
            "Provide a brief explanation for your decision and then state your verdict."
        )
        user_message = f"Prompt: {prompt}\n\nResponses:\n"
        for idx, response in enumerate(responses, 1):
            user_message += f"{idx}. {response}\n\n"

        result = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=JudgmentResult,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        if regenerate:
            if verbose:
                logger.warning(f"Response is not accurate. Regenerating response.")

            system_message = (
                "Based on the judgment provided, generate an optimized response "
                "that addresses the prompt more effectively than the original responses."
            )
            user_message = f"Original prompt: {prompt}\n\nJudgment: {result.explanation}\n\nGenerate an optimized response:"

            regenerated = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                response_model=RegeneratedResponse,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
            )
            result = regenerated

    elif process == "validate":
        if not schema:
            raise ValueError("Schema is required for validation.")

        system_message = (
            "You are a validation expert. Your task is to determine if the given response "
            "matches the provided schema or instructions. Provide a detailed explanation "
            "of your validation process and state whether the response is valid or not."
        )
        user_message = f"Prompt: {prompt}\n\nResponse: {responses[0]}\n\nSchema/Instructions: {schema}"

        result = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=ValidationResult,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        if regenerate and not result.is_valid:
            if verbose:
                logger.warning(f"Response is not valid. Regenerating response.")

            system_message = (
                "Based on the validation result, generate a new response that "
                "correctly adheres to the given schema or instructions."
            )
            user_message = f"Original prompt: {prompt}\n\nSchema/Instructions: {schema}\n\nGenerate a valid response:"

            regenerated = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                response_model=RegeneratedResponse,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
            )
            result = regenerated

    elif process == "fact_check":
        if responses is None:
            responses = [prompt]  # Use the prompt as the response for fact-checking
        elif len(responses) != 1:
            raise ValueError("Fact-check requires exactly one response.")

        system_message = (
            "You are a fact-checking expert. Your task is to determine if the given response "
            "is accurate based on the prompt and your knowledge. Provide a detailed explanation "
            "of your fact-checking process, state whether the response is accurate or not, "
            "and provide a confidence score between 0.0 and 1.0."
        )
        user_message = f"Prompt: {prompt}\n\nResponse to fact-check: {responses[0]}"
        if schema:
            user_message += f"\n\nAdditional fact-checking guidelines: {schema}"

        result = completion_client.completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model=model,
            response_model=FactCheckResult,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        if regenerate and not result.is_accurate:
            if verbose:
                logger.warning(f"Response is not accurate. Regenerating response.")

            system_message = (
                "Based on the fact-check result, generate a new response that "
                "is accurate and addresses the original prompt correctly."
            )
            user_message = f"Original prompt: {prompt}\n\nFact-check result: {result.explanation}\n\nGenerate an accurate response:"

            regenerated = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                response_model=RegeneratedResponse,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
            )
            result = regenerated

    else:
        raise ValueError(
            "Invalid process. Choose 'accuracy', 'validate', or 'fact_check'."
        )

    # Add guardrails check after the main process
    if guardrails:
        guardrails_result = check_guardrails(
            prompt,
            result,
            guardrails,
            completion_client,
            model,
            mode,
            max_retries,
            temperature,
            verbose,
        )
        if not guardrails_result.passed:
            if verbose:
                logger.warning(f"Response violates guardrails. Regenerating response.")
            result = regenerate_response(
                prompt,
                guardrails_result.explanation,
                completion_client,
                model,
                mode,
                max_retries,
                temperature,
            )

    return result


def check_guardrails(
    prompt,
    result,
    guardrails,
    completion_client,
    model,
    mode,
    max_retries,
    temperature,
    verbose,
):
    if isinstance(guardrails, str):
        guardrails = [guardrails]

    guardrails_prompt = "\n".join(guardrails)

    system_message = (
        "You are a content moderator. Your task is to determine if the given response "
        "violates any of the specified guardrails. Provide a detailed explanation "
        "of your evaluation process and state whether the response passes all guardrails or not."
    )
    user_message = (
        f"Prompt: {prompt}\n\nResponse: {result}\n\nGuardrails:\n{guardrails_prompt}"
    )

    guardrails_result = completion_client.completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_model=GuardrailsResult,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
    )

    return guardrails_result


def regenerate_response(
    prompt, explanation, completion_client, model, mode, max_retries, temperature
):
    system_message = (
        "Based on the guardrails violation, generate a new response that "
        "addresses the original prompt while adhering to all specified guardrails."
    )
    user_message = f"Original prompt: {prompt}\n\nGuardrails violation: {explanation}\n\nGenerate a compliant response:"

    regenerated = completion_client.completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_model=RegeneratedResponse,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
    )
    return regenerated


class GuardrailsResult(BaseModel):
    passed: bool
    explanation: str


if __name__ == "__main__":
    # Example usage
    prompt = "Explain the concept of quantum entanglement."
    responses = [
        "Quantum entanglement is a phenomenon where two particles become interconnected and their quantum states cannot be described independently.",
        "Quantum entanglement is when particles are really close to each other and move in the same way.",
    ]

    # Accuracy judgment
    result = judge(prompt, responses, process="accuracy", verbose=True)
    print(
        f"Accuracy Judgment:\nExplanation: {result.explanation}\nVerdict: {result.verdict}"
    )

    # Validation
    schema = "The response should include: 1) Definition of quantum entanglement, 2) Its importance in quantum mechanics, 3) An example or application."
    result = judge(
        prompt, [responses[0]], process="validate", schema=schema, verbose=True
    )
    print(
        f"\nValidation Result:\nIs Valid: {result.is_valid}\nExplanation: {result.explanation}"
    )

    # Fact-check
    fact_check_response = "Quantum entanglement occurs when two particles are separated by a large distance but still instantaneously affect each other's quantum states."
    result = judge(prompt, [fact_check_response], process="fact_check", verbose=True)
    print(
        f"\nFact-Check Result:\nIs Accurate: {result.is_accurate}\nExplanation: {result.explanation}\nConfidence: {result.confidence}"
    )

    # Regeneration
    result = judge(prompt, responses, process="accuracy", regenerate=True, verbose=True)
    print(f"\nRegenerated Response:\n{result.response}")
