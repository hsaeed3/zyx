from typing import Optional, Literal, Union


def format_messages(
    messages: Union[str, list[dict]] = None,
    verbose: Optional[bool] = False,
    type: Optional[Literal["user", "system", "assistant"]] = "user",
) -> list[dict]:
    """Formats the messages into a list of dictionaries.

    Args:
        messages: Union[str, list[dict]]: The messages to format.
        verbose: bool: Whether to log the formatting process.

    Returns:
        list[dict]: The formatted messages.
    """

    try:
        if isinstance(messages, str):
            if verbose:
                print(f"Converting string to message format.")

            return [{"role": type, "content": messages}]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            if verbose:
                print(f"Messages are in the correct format.")

            return messages
        else:
            raise ValueError("Invalid message format")
    except Exception as e:
        print(f"Error formatting messages: {e}")
        return []


def does_system_prompt_exist(messages: list[dict]) -> bool:
    """Simple boolean check to see if a system prompt exists in the messages.

    Args:
        messages: list[dict]: The messages to check.

    Returns:
        bool: True if a system prompt exists, False otherwise.
    """

    return any(message.get("role") == "system" for message in messages)


def swap_system_prompt(
    system_prompt: dict = None, messages: Union[str, list[dict[str, str]]] = None
):
    """Swaps the system prompt with the system_prompt.

    Args:
        system_prompt: dict: The system prompt to swap.
        messages: Union[str, list[dict[str, str]]]: The messages to swap.

    Returns:
        list[dict[str, str]]: The messages with the system prompt swapped.
    """

    messages = format_messages(messages)

    for message in messages:
        # Check if a system message exists
        if message.get("role") == "system":
            # If a system message exists, swap it with the system_prompt
            message = system_prompt
            # Move the system_prompt to the beginning of the list
            messages.insert(0, message)
            # Remove the system_prompt from its original position
            messages.remove(message)
            break

        else:
            # If no system message exists, add the system_prompt to the beginning of the list
            messages.insert(0, system_prompt)
            break

    # Remove any duplicate system messages
    while len([message for message in messages if message.get("role") == "system"]) > 1:
        messages.pop()

    return messages


def repair_messages(
    messages: list[dict], verbose: Optional[bool] = False
) -> list[dict]:
    """
    Repairs the messages by performing quick logic steps. Does not
    raise exceptions, attempts to fix the messages in a best effort manner.

    Args:
        messages: list[dict]: The messages to repair.
        verbose: Optional[bool]: Whether to log the repair process.

    Returns:
        list[dict]: The repaired messages.
    """

    # Ensure no item in the list is a nested list.
    if any(isinstance(message, list) for message in messages):
        messages = [item for sublist in messages for item in sublist]
        if verbose:
            print(f"Detected nested lists and flattened the list.")

    # Ensure messages are in the role user -> role assistant order &
    # repair order if items are mixmatched
    for i in range(len(messages) - 1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            if (
                not isinstance(messages[i + 1], dict)
                or messages[i + 1].get("role") != "user"
            ):
                messages[i + 1] = {"role": "user", "content": ""}
                if verbose:
                    print(f"Detected a mixmatch in message order, repaired order.")
        elif isinstance(messages[i], dict) and messages[i].get("role") == "user":
            if (
                not isinstance(messages[i + 1], dict)
                or messages[i + 1].get("role") != "assistant"
            ):
                messages[i + 1] = {"role": "assistant", "content": ""}
                if verbose:
                    print(f"Detected a mixmatch in message order, repaired order.")

    return messages


def add_messages(
    inputs: Union[str, list[dict], dict] = None,
    messages: list[dict] = None,
    type: Optional[Literal["user", "system", "assistant"]] = "user",
    verbose: Optional[bool] = False,
) -> list[dict]:
    """
    Adds a message to the thread, based on the type of message to add; and
    after performing some basic checks.

    Args:
        inputs: Union[str, list[dict], dict]: The messages to add.
        messages: list[dict]: The existing messages.
        type: Optional[Literal["user", "system", "assistant"]]: The type of message to add.
        verbose: Optional[bool]: Whether to log the addition of the message.

    Returns:
        list[dict]: The messages with the added message(s).
    """

    if isinstance(inputs, str):
        formatted_message = format_messages(messages=inputs, verbose=verbose, type=type)

        messages.extend(formatted_message)

    elif isinstance(inputs, dict):
        messages.append(inputs)

    elif isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict):
                messages.append(item)
            else:
                if verbose:
                    print(f"Skipping invalid message format: {item}")

    return repair_messages(messages, verbose)


if __name__ == "__main__":
    sample_text = "Hello, how are you?"
    sample_thread = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"},
    ]

    broken_thread = [
        [{"role": "user", "content": "Hello, how are you?"}],
        {"role": "assistant", "content": "I am fine, thank you!"},
    ]

    print(format_messages(sample_text))

    print(does_system_prompt_exist(sample_thread))

    print(
        swap_system_prompt(
            system_prompt={"role": "system", "content": "You are a helpful assistant."},
            messages=sample_thread,
        )
    )

    print(repair_messages(broken_thread))

    print(
        add_messages(
            inputs=sample_text, messages=sample_thread, type="user", verbose=True
        )
    )
