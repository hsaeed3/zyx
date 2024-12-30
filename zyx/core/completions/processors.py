from __future__ import annotations

"""
zyx.core.completions.processors

This module provides utility functions for the completions runnable.
"""

# [Imports]
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence
from pydantic import BaseModel

from ...types.completions import Message, MessageRole, Image
from zyx.core import utils


# ==============================================================
# [Main Class]
# ==============================================================

class CompletionsProcessors:
    
    """
    Main class for all completions processors.
    """
    
    # [Message Processor]
    messages : MessagesProcessor
    
    # [Prompting Processor]


# ==============================================================
# [Message Processor]
# ==============================================================

class MessagesProcessor:
    
    """
    Messages processors.
    """

    @staticmethod
    def create_message(
        content: Any,
        role: MessageRole = "user",
        images: Optional[Union[Image, Sequence[Image], Path, str]] = None
    ) -> Message:
        """
        Creates a single message based on the provided content & role.
        Additionally takes in images for multimodal models.
        
        Args:
            content (Any): The content of the message.
            role (MessageRole): The role of the message.
            images (Optional[Union[Image, Sequence[Image], Path, str]]): The images to add to the message.
                Can be:
                - Path to image file
                - URL string
                - Image object
                - List of any of the above
                
        Returns:
            Message: The created message.
        """    
        # Handle image case
        if images: 
            if not role == "user":
                raise utils.ZyxException("Images can only be added to user messages!")
            
            # Convert to list if single item
            if not isinstance(images, (list, tuple)):
                images = [images]
                
            # Convert each image to Image object
            processed_images = []
            for img in images:
                if isinstance(img, Image):
                    processed_images.append(img)
                else:
                    # Path, str (URL or base64), or bytes will be handled by Image model
                    processed_images.append(Image(value=img))
                    
            if utils.zyx_debug:
                utils.logger.debug(
                    f"Processed images for {role} message: {processed_images}"
                )
            
            images = processed_images

        # Handle non-string content case
        if not isinstance(content, str):
            content = json.dumps(content)
            
        if utils.zyx_debug:
            utils.logger.debug(
                f"Created [plum3]{role}[/plum3] message with content [italic dim]{content}[/italic dim]"
            )
            
        # Return
        return Message(
            role=role,
            content=content,
            images=images
        )


    @staticmethod
    def validate_message(
        message : Union[Message, BaseModel, Dict[str, Any]]
    ) -> Message:
        """
        Validates a single message. Currently only runs a simple check to ensure 'role' is present.
        """
        if isinstance(message, BaseModel):
            message = message.model_dump()
            
        if "role" not in message:
            raise utils.ZyxException(f"Invalid message provided [bold red]no role field found[/bold red], `{message}`.")
        
        if not isinstance(message, Message):
            return Message(**message)
        return message


    @staticmethod
    def format_messages_input(
        messages : Union[
            # string messages are converted to one 'role' : 'user' message
            str,
            # standard single message
            Union[Message, BaseModel, Dict[str, Any]],
            # list of messages
            Sequence[Union[Message, BaseModel, Dict[str, Any]]]
        ]
    ) -> List[Message]:
        """
        Formats input messages into a valid list of OpenAI spec chat completion messages.
        
        Args:
            messages (Union[str, Union[Message, BaseModel, Dict[str, Any]], Sequence[Union[Message, BaseModel, Dict[str, Any]]]]): The messages to format.
            
        Returns:
            List[Message]: The formatted messages.
        """
        # clear string case immediately
        if isinstance(messages, str):
            
            if utils.zyx_debug:
                utils.logger.debug(
                    f"Formatting single string into message thread: [italic dim]{messages}[/italic dim]"
                )
            
            return [MessagesProcessor.create_message(content = messages)]
        
        if not isinstance(messages, list):
            messages = [messages]
            
        for message in messages:
            message = MessagesProcessor.validate_message(message)
            
        if utils.zyx_debug:
            utils.logger.debug(
                f"Formatted {len(messages)} messages into message thread."
            )
            
        return messages
    
    
    @staticmethod
    def format_system_message_inside_thread(
        messages: List[Union[Message, Dict[str, Any], BaseModel]]
    ) -> List[Message]:
        """
        Validates system instructions in a thread of messages. 
        
        Pipeline:
        - First checks if a system message or messages are present in the thread
        - Merges system content if multiple messages are present and ensures it is at the beginning of the thread
        - If no system message is present, passes
        
        Args:
            messages: List of messages to process. Can be Message objects, dictionaries, or BaseModels.
            
        Returns:
            List[Message]: Processed list of messages with system messages properly formatted.
        """
        # Convert all messages to Message objects first
        messages = [MessagesProcessor.validate_message(msg) for msg in messages]
        
        # [Check for System Message]
        system_messages = [message for message in messages if message.role == "system"]
        
        if not system_messages:
            
            if utils.zyx_debug:
                utils.logger.debug("No system message found in thread, returning as is.")
            
            return messages
        
        if len(system_messages) > 1:
            # Create merged system message content
            system_message_content = "\n".join([message.content for message in system_messages])
            system_message = MessagesProcessor.create_message(content=system_message_content, role="system")
            
            # Remove all previous system messages in thread
            messages = [message for message in messages if message.role != "system"]
            
            # Add merged system message to the beginning of the thread
            messages.insert(0, system_message)
            
            if utils.zyx_debug:
                utils.logger.debug(
                    f"Merged {len(system_messages)} system messages into one at the beginning of the thread."
                )
            
        elif len(system_messages) == 1:
            # Validate system message is at the beginning of the thread
            if not messages[0] == system_messages[0]:
                # Remove the existing system message
                messages = [message for message in messages if message.role != "system"]
                # Add the new system message to the beginning of the thread
                messages.insert(0, system_messages[0])
                
                if utils.zyx_debug:
                    utils.logger.debug(
                        f"Validated system message position in thread."
                    )

        return messages
        

# ==============================================================
# [Prompting Processor]
# ==============================================================

class PromptingProcessor:
    
    """
    Prompting utility & processors.
    """
    
    @staticmethod
    def create_string_context_from_object(
        context: Union[str, BaseModel, Any]
    ) -> str:
        """
        Creates a context string from objects.

        Args:
            context: The context object to convert to string. Can be:
                - String (returned as-is)
                - Pydantic BaseModel class (converted to JSON schema)
                - Pydantic BaseModel instance (converted to JSON)
                - Other objects (converted to JSON string)

        Returns:
            str: The context as a string.

        Raises:
            ZyxException: If the object cannot be converted to a string context.
        """
        # Create context string from pydantic models
        if isinstance(context, type) and issubclass(context, BaseModel):
            try:
                context = context.model_json_schema()
            except Exception as e:
                raise utils.ZyxException(f"Failed to get JSON schema from model class {utils.Styles.module(context)}: {e}")
        elif isinstance(context, BaseModel):
            try:
                context = context.model_dump()
            except Exception as e:
                raise utils.ZyxException(f"Failed to dump pydantic model {utils.Styles.module(context)} into dict: {e}")

        # Convert to JSON string if not already
        if not isinstance(context, str):
            try:
                context = json.dumps(context)
            except Exception as e:
                raise utils.ZyxException(f"Failed to convert object {utils.Styles.module(context)} to JSON string context: {e}")
            
        if utils.zyx_debug:
            utils.logger.debug(
                f"Created context string from object {utils.Styles.module(context)}."
            )
            
        return context
    
    
    @staticmethod
    def add_system_context_to_thread(
        context : str,
        messages : List[Union[Message, Dict[str, Any], BaseModel]],
    ) -> List[Message]:
        """
        Adds system context to a thread of messages.
        """
    
        # Determine if system message is present
        if any(message.role == "system" for message in messages):
            # Format thread to validate system message position
            messages = MessagesProcessor.format_system_message_inside_thread(messages)

            # Build context into system message
            system_content = messages[0]['content']
            system_content = f"{system_content}\n\n{context}"
            messages[0]['content'] = system_content
            
        else:
            # Create new system message
            messages.insert(0, MessagesProcessor.create_message(content = context, role = "system"))
            if utils.zyx_debug:
                utils.logger.debug(
                    f"Added system context to thread as a new system message."
                )
        # Return
        return messages