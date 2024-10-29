from typing import Union, Optional, Literal


class MessagesUtils:


    @staticmethod
    def format_messages(
        messages: Union[str, list[dict]] = None,
        verbose: Optional[bool] = True,
        type: Optional[Literal["user", "system", "assistant"]] = "user",
    ) -> list[dict]:
        """Formats the messages into a list of dictionaries and repairs invalid formats.

        Args:
            messages: Union[str, list[dict]]: The messages to format.
            verbose: bool: Whether to log the formatting process.
            type: Literal["user", "system", "assistant"]: The role type for string messages.

        Returns:
            list[dict]: The formatted messages with repaired formats.
        """

        try:
            # Handle None or empty string case
            if messages is None or messages == "":
                if verbose:
                    print("No messages provided, returning empty list.")
                return []

            if isinstance(messages, str):
                if verbose:
                    print(f"Converting string to message format.")
                return [{"role": type, "content": messages}]
                
            elif isinstance(messages, list):
                formatted_messages = []
                for m in messages:
                    if isinstance(m, str):
                        formatted_messages.append({"role": type, "content": m})
                    elif isinstance(m, dict):
                        # Accept any valid message dict as-is
                        if "role" in m and "content" in m:
                            # Ensure content is string and has valid JSON formatting
                            content = m["content"]
                            if not isinstance(content, str):
                                content = str(content)
                            formatted_messages.append({
                                "role": m["role"],
                                "content": content,
                                "refusal": m.get("refusal"),
                                "audio": m.get("audio"),
                                "function_call": m.get("function_call"),
                                "tool_calls": m.get("tool_calls")
                            })
                        else:
                            # Only repair if missing required fields
                            role = m.get("role", type)
                            content = m.get("content", "")
                            if not isinstance(content, str):
                                content = str(content)
                            formatted_messages.append({
                                "role": role,
                                "content": content,
                                "refusal": None,
                                "audio": None,
                                "function_call": None,
                                "tool_calls": None
                            })
                    else:
                        if verbose:
                            print(f"Skipping invalid message format: {m}")
                        continue
                        
                if verbose and formatted_messages:
                    print("Messages have been formatted and repaired.")
                return formatted_messages
                
            else:
                if verbose:
                    return formatted_messages

        except Exception as e:
            print(f"Error formatting messages: {e}")
            return []

    @staticmethod
    def does_system_prompt_exist(messages: list[dict]) -> bool:
        """Simple boolean check to see if a system prompt exists in the messages.

        Args:
            messages: list[dict]: The messages to check.

        Returns:
            bool: True if a system prompt exists, False otherwise.
        """

        return any(message.get("role") == "system" for message in messages)

    @staticmethod
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
        # Handle None cases
        if system_prompt is None or messages is None:
            return []

        # Create new messages list starting with system prompt
        new_messages = [system_prompt]

        # Add all messages from original thread, preserving order
        # Just skip any existing system messages
        for message in messages:
            if not message.get("role") == "system":
                new_messages.append(message)

        return new_messages

    @staticmethod
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

    @staticmethod
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
            formatted_message = MessagesUtils.format_messages(
                messages=inputs, verbose=verbose, type=type
            )

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

        return MessagesUtils.repair_messages(messages, verbose)
