from typing import Literal, Union


def format_messages(messages: Union[str, list[dict]] = None) -> list[dict]:
    try:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            return messages
        else:
            raise ValueError("Invalid message format")
    except Exception as e:
        print(f"Error formatting messages: {e}")
        return []


def does_system_prompt_exist(messages: list[dict]) -> bool:
    return any(message.get("role") == "system" for message in messages)


def swap_system_prompt(
    system_prompt: dict = None, messages: Union[str, list[dict[str, str]]] = None
):
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
