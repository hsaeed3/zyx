from typing import List, Optional, Literal
from ...types import client as clienttypes


def verifier(
    prompt: str,
    assistant_responses: List[str],
    user_question: Optional[str] = None,
    model: str = "gpt-4o-mini",
    client: Literal["openai", "litellm"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: clienttypes.InstructorMode = "chat",
    max_retries: int = 3,
    temperature: float = 0.0,
    run_tools: Optional[bool] = True,
    tools: Optional[List[clienttypes.ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    verbose: Optional[bool] = False,
    **kwargs,
) -> str:
    """
    Compares two assistant responses and determines which one is better.

    Example:
    ```python
    prompt = "Explain the significance of the Emancipation Proclamation in American history."

    assistant_response_a = (
        "The Emancipation Proclamation, issued by President Abraham Lincoln on January 1, 1863, "
        "declared that all enslaved people in Confederate-held territory were free. This was a crucial "
        "turning point in the Civil War, transforming the fight to preserve the nation into a battle "
        "for human freedom. It also allowed African Americans to join the Union Army, bolstering its "
        "numbers and morale."
    )

    assistant_response_b = (
        "The Emancipation Proclamation was an order by President Lincoln during the Civil War. "
        "It said that slaves in the Confederate states were free. This was important because it changed "
        "the nature of the war and led to the freedom of many slaves."
    )

    judgment = verifier(
        prompt=prompt,
        assistant_responses=[assistant_response_a, assistant_response_b],
        api_key="YOUR_API_KEY",
        verbose=True,
    )

    print("\nFinal Judgment from verifier function:")
    print(judgment)
    ```
    
    Args:
        prompt (str): The initial prompt or user question.
        assistant_responses (List[str]): A list containing two responses from different assistants/models.
        user_question (Optional[str]): (Optional) The user question if different from the prompt.
        model (str): The model to use for the verifier.
        client (Literal["openai", "litellm"]): The client to use for the verifier.
        api_key (Optional[str]): The API key to use for the verifier.
        base_url (Optional[str]): The base URL to use for the verifier.
        mode (clienttypes.InstructorMode): The mode to use for the verifier.
        max_retries (int): The maximum number of retries to use for the verifier.
        temperature (float): The temperature to use for the verifier.
        run_tools (Optional[bool]): Whether to run tools for the verifier.
        tools (Optional[List[clienttypes.ToolType]]): The tools to use for the verifier.
        parallel_tool_calls (Optional[bool]): Whether to run tools in parallel for the verifier.
        tool_choice (Optional[Literal["none", "auto", "required"]]): The tool choice to use for the verifier.
        verbose (Optional[bool]): Whether to print the verifier's output.
        **kwargs: Additional keyword arguments to pass to the verifier.
    
    Returns:
        str: The judgment of the verifier.
    """

    from ..client import completion
    if user_question is None:
        user_question = prompt

    if len(assistant_responses) != 2:
        raise ValueError("Exactly two assistant responses are required for comparison.")

    # Prepare the messages for the LLM judge
    messages = [
        {
            "role": "system",
            "content": (
                "Please act as an impartial judge and evaluate the quality of the responses "
                "provided by two AI assistants to the user question displayed below. You should "
                "choose the assistant that follows the user's instructions and answers the user's "
                "question better. Your evaluation should consider factors such as helpfulness, "
                "relevance, accuracy, depth, creativity, and level of detail of their responses. "
                "Begin your evaluation by comparing the two responses and provide a short explanation. "
                "Avoid any position biases and ensure that the order in which the responses were "
                "presented does not influence your decision. Do not allow the length of the responses "
                "to influence your evaluation. Do not favor certain names of the assistants. "
                "Be as objective as possible. After providing your explanation, output your final "
                "verdict by strictly following this format: '[[A]]' if assistant A is better, "
                "'[[B]]' if assistant B is better, and '[[C]]' for a tie."
            ),
        },
        {"role": "user", "content": f"[User Question]\n{user_question}"},
        {
            "role": "assistant",
            "content": (
                "[The Start of Assistant A's Answer]\n"
                f"{assistant_responses[0]}\n"
                "[The End of Assistant A's Answer]\n"
                "[The Start of Assistant B's Answer]\n"
                f"{assistant_responses[1]}\n"
                "[The End of Assistant B's Answer]"
            ),
        },
    ]

    # Call the completion function to get the judgment
    response = completion(
        messages=messages,
        model=model,
        client=client,
        api_key=api_key,
        base_url=base_url,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
        run_tools=run_tools,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        verbose=verbose,
        **kwargs,
    )

    # Extract the content from the response
    judgment = response.choices[0].message["content"]

    if verbose:
        print(f"Verifier Judgment:\n{judgment}")

    return judgment


class Judge:
    """
    Implements the LLM-as-a-judge concept to evaluate and compare responses from language models.
    Handles multiple inputs and provides additional functionalities.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Literal["openai", "litellm"] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: clienttypes.InstructorMode = "chat",
        max_retries: int = 3,
        temperature: float = 0.0,
        run_tools: Optional[bool] = True,
        tools: Optional[List[clienttypes.ToolType]] = None,
        parallel_tool_calls: Optional[bool] = False,
        tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
        verbose: Optional[bool] = False,
        **kwargs,
    ):
        """
        Initializes the Judge class with the specified parameters.

        Args:
            model (str): The model to use for the judge.
            client (Literal["openai", "litellm"]): The client to use for the judge.
            api_key (Optional[str]): The API key to use for the judge.
            base_url (Optional[str]): The base URL to use for the judge.
            mode (clienttypes.InstructorMode): The mode to use for the judge.
            max_retries (int): The maximum number of retries to use for the judge.
            temperature (float): The temperature to use for the judge.
            run_tools (Optional[bool]): Whether to run tools for the judge.
            tools (Optional[List[clienttypes.ToolType]]): The tools to use for the judge.
            parallel_tool_calls (Optional[bool]): Whether to run tools in parallel for the judge.
            tool_choice (Optional[Literal["none", "auto", "required"]]): The tool choice to use for the judge.
            verbose (Optional[bool]): Whether to print the judge's output.
            **kwargs: Additional keyword arguments to pass to the judge.
        """

        from ..client import completion

        self.completion = completion

        self.model = model
        self.client = client
        self.api_key = api_key
        self.base_url = base_url
        self.mode = mode
        self.max_retries = max_retries
        self.temperature = temperature
        self.run_tools = run_tools
        self.tools = tools
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_choice = tool_choice
        self.verbose = verbose
        self.kwargs = kwargs

    def judge_pair(
        self,
        prompt: str,
        assistant_responses: List[str],
        user_question: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Compares two assistant responses and determines which one is better.

        Args:
            prompt (str): The initial prompt or user question.
            assistant_responses (List[str]): A list containing two responses from different assistants/models.
            user_question (Optional[str]): (Optional) The user question if different from the prompt.
            **kwargs: Additional keyword arguments to pass to the judge.
        
        Returns:
            str: The judgment of the judge.
        """
        return verifier(
            prompt=prompt,
            assistant_responses=assistant_responses,
            user_question=user_question,
            model=self.model,
            client=self.client,
            api_key=self.api_key,
            base_url=self.base_url,
            mode=self.mode,
            max_retries=self.max_retries,
            temperature=self.temperature,
            run_tools=self.run_tools,
            tools=self.tools,
            parallel_tool_calls=self.parallel_tool_calls,
            tool_choice=self.tool_choice,
            verbose=self.verbose,
            **{**self.kwargs, **kwargs},
        )

    def judge_multiple(
        self,
        prompt: str,
        assistant_responses: List[str],
        user_question: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Compares multiple assistant responses and ranks them.

        Args:
            prompt (str): The initial prompt or user question.
            assistant_responses (List[str]): A list of responses from different assistants/models.
            user_question (Optional[str]): (Optional) The user question if different from the prompt.
            **kwargs: Additional keyword arguments to pass to the judge.
        
        Returns:
            str: The judgment of the judge.
        """
        if user_question is None:
            user_question = prompt

        num_responses = len(assistant_responses)
        if num_responses < 2:
            raise ValueError("At least two assistant responses are required for comparison.")

        # Prepare the messages for the LLM judge
        assistant_content = ""
        for idx, response in enumerate(assistant_responses):
            assistant_content += (
                f"[The Start of Assistant {chr(65 + idx)}'s Answer]\n"
                f"{response}\n"
                f"[The End of Assistant {chr(65 + idx)}'s Answer]\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Please act as an impartial judge and evaluate the quality of the responses "
                    "provided by multiple AI assistants to the user question displayed below. You should "
                    "rank the assistants based on how well they follow the user's instructions and answer "
                    "the user's question. Your evaluation should consider factors such as helpfulness, "
                    "relevance, accuracy, depth, creativity, and level of detail of their responses. "
                    "Begin your evaluation by comparing the responses and provide a short explanation for "
                    "each. Avoid any position biases and ensure that the order in which the responses were "
                    "presented does not influence your decision. Do not allow the length of the responses "
                    "to influence your evaluation. Do not favor certain names of the assistants. "
                    "Be as objective as possible. After providing your explanation, output your final "
                    "verdict by strictly following this format: '[[Rank]]' followed by the ranking of the assistants "
                    "from best to worst (e.g., [[Rank]] A > B > C)."
                ),
            },
            {"role": "user", "content": f"[User Question]\n{user_question}"},
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]

        # Call the completion function to get the judgment
        response = self.completion(
            messages=messages,
            model=self.model,
            client=self.client,
            api_key=self.api_key,
            base_url=self.base_url,
            mode=self.mode,
            max_retries=self.max_retries,
            temperature=self.temperature,
            run_tools=self.run_tools,
            tools=self.tools,
            parallel_tool_calls=self.parallel_tool_calls,
            tool_choice=self.tool_choice,
            verbose=self.verbose,
            **{**self.kwargs, **kwargs},
        )

        # Extract the content from the response
        judgment = response.choices[0].message["content"]

        if self.verbose:
            print(f"Judge's Evaluation:\n{judgment}")

        return judgment

    def batch_judge(
        self,
        prompts: List[str],
        assistant_responses_list: List[List[str]],
        user_questions: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Evaluates multiple prompts and assistant responses in batch.

        Args:
            prompts (List[str]): A list of prompts to evaluate.
            assistant_responses_list (List[List[str]]): A list of lists of responses from different assistants/models.
            user_questions (Optional[List[str]]): (Optional) A list of user questions if different from the prompts.
            **kwargs: Additional keyword arguments to pass to the judge.
        
        Returns:
            List[str]: A list of judgments for each prompt.
        """
        if user_questions is None:
            user_questions = prompts

        if not (len(prompts) == len(assistant_responses_list) == len(user_questions)):
            raise ValueError("prompts, assistant_responses_list, and user_questions must have the same length.")

        judgments = []
        for prompt, responses, question in zip(prompts, assistant_responses_list, user_questions):
            judgment = self.judge_multiple(
                prompt=prompt,
                assistant_responses=responses,
                user_question=question,
                **kwargs,
            )
            judgments.append(judgment)
            if self.verbose:
                print(f"Judgment for prompt '{prompt}':\n{judgment}\n")

        return judgments

# Example usage
if __name__ == "__main__":
    # Use the standalone verifier function for a single use case
    prompt = "Explain the significance of the Emancipation Proclamation in American history."

    assistant_response_a = (
        "The Emancipation Proclamation, issued by President Abraham Lincoln on January 1, 1863, "
        "declared that all enslaved people in Confederate-held territory were free. This was a crucial "
        "turning point in the Civil War, transforming the fight to preserve the nation into a battle "
        "for human freedom. It also allowed African Americans to join the Union Army, bolstering its "
        "numbers and morale."
    )

    assistant_response_b = (
        "The Emancipation Proclamation was an order by President Lincoln during the Civil War. "
        "It said that slaves in the Confederate states were free. This was important because it changed "
        "the nature of the war and led to the freedom of many slaves."
    )

    judgment = verifier(
        prompt=prompt,
        assistant_responses=[assistant_response_a, assistant_response_b],
        api_key="YOUR_API_KEY",
        verbose=True,
    )

    print("\nFinal Judgment from verifier function:")
    print(judgment)

    # Use the Judge class to handle multiple inputs
    judge = Judge(
        model="gpt-4o-mini",
        client="openai",
        api_key="YOUR_API_KEY",
        verbose=True,
    )

    # Comparing two responses using the judge_pair method
    pair_judgment = judge.judge_pair(
        prompt=prompt,
        assistant_responses=[assistant_response_a, assistant_response_b],
    )

    print("\nFinal Judgment from judge_pair method:")
    print(pair_judgment)

    # Comparing multiple responses using the judge_multiple method
    assistant_response_c = (
        "The Emancipation Proclamation, signed by President Lincoln in 1863, declared the freedom of all slaves in Confederate territory. "
        "This shifted the Civil War's focus to include the abolition of slavery as a Union goal, discouraging European powers from supporting "
        "the Confederacy and allowing Black men to serve in the Union Army and Navy, which bolstered Union forces."
    )

    multiple_judgment = judge.judge_multiple(
        prompt=prompt,
        assistant_responses=[assistant_response_a, assistant_response_b, assistant_response_c],
    )

    print("\nFinal Judgment from judge_multiple method:")
    print(multiple_judgment)

    # Batch judging multiple prompts
    prompts = [
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
    ]

    assistant_responses_list = [
        [
            "Climate change is primarily caused by the increase in greenhouse gases like carbon dioxide in the atmosphere due to human activities such as burning fossil fuels, deforestation, and industrial processes.",
            "The main causes of climate change are natural phenomena like volcanic eruptions and variations in solar radiation."
        ],
        [
            "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
            "Photosynthesis is how plants make food."
        ]
    ]

    batch_judgments = judge.batch_judge(
        prompts=prompts,
        assistant_responses_list=assistant_responses_list,
    )

    print("\nFinal Judgments from batch_judge method:")
    for idx, judgment in enumerate(batch_judgments):
        print(f"Judgment for prompt '{prompts[idx]}':\n{judgment}\n")
