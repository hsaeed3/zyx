from typing import List, Optional, Literal, Union, Type
from pydantic import BaseModel, Field
from ....lib.types.document import Document
from ....client import Client, InstructorMode
from ....lib.utils.logger import get_logger
from ..agents.judge import judge, ValidationResult
from ...ext.multimodal import OPENAI_TTS_VOICES, OPENAI_TTS_MODELS, audio
from ..base.classify import classify

logger = get_logger("conversation")


class Character(BaseModel):
    name: str
    personality: str
    knowledge: Optional[str] = None
    voice: Optional[OPENAI_TTS_VOICES] = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    audio_file: Optional[str] = None


class Conversation(BaseModel):
    messages: List[Message]
    audio_file: Optional[str] = None


class EndConversation(BaseModel):
    should_end: bool
    explanation: Optional[str] = None
    confidence: Optional[float] = None


class ConversationEndCheck(BaseModel):
    should_end: bool
    explanation: Optional[str] = None


import tempfile
import os
from pydub import AudioSegment
from typing import List, Optional, Literal, Union, Type
from pydantic import BaseModel, Field
from ....lib.types.document import Document
from ....client import Client, InstructorMode
from ....lib.utils.logger import get_logger
from ..agents.judge import judge, ValidationResult
from ...ext.multimodal import OPENAI_TTS_VOICES, OPENAI_TTS_MODELS, audio

logger = get_logger("conversation")


class Character(BaseModel):
    name: str
    personality: str
    knowledge: Optional[str] = None
    voice: Optional[OPENAI_TTS_VOICES] = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    audio_file: Optional[str] = None


class Conversation(BaseModel):
    messages: List[Message]
    audio_file: Optional[str] = None


class EndConversation(BaseModel):
    should_end: bool
    explanation: Optional[str] = None
    confidence: Optional[float] = None


class ConversationEndCheck(BaseModel):
    should_end: bool
    explanation: Optional[str] = None


def conversation(
    instructions: Union[str, Document],
    characters: List[Character],
    validator: Optional[Union[str, dict]] = None,
    min_turns: int = 5,
    max_turns: int = 20,
    end_criteria: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    mode: InstructorMode = "markdown_json_mode",
    max_retries: int = 3,
    organization: Optional[str] = None,
    client: Optional[Literal["openai", "litellm"]] = None,
    verbose: bool = False,
    generate_audio: bool = False,
    audio_model: OPENAI_TTS_MODELS = "tts-1",
    audio_output_file: Optional[str] = None,
) -> Conversation:
    """
    Generate a conversation between characters based on given instructions or a Document object, with optional validator.

    Example:
        ```python
        from zyx import Document
        from zyx.resources.completions.agents.conversation import conversation, Character

        doc = Document(content="The impact of AI on job markets", metadata={"type": "research_paper"})
        result = conversation(
            instructions=doc,
            characters=[
                Character(name="AI Researcher", personality="Optimistic about AI's potential", voice="nova"),
                Character(name="Labor Economist", personality="Concerned about job displacement", voice="onyx"),
                Character(name="Podcast Host", personality="Neutral moderator", voice="echo")
            ],
            min_turns=10,
            max_turns=15,
            end_criteria="The podcast should conclude with final thoughts from both guests",
            verbose=True,
            generate_audio=True,
            audio_output_file="ai_job_market_podcast.mp3"
        )
        print(result.messages)
        ```

    Args:
        instructions (Union[str, Document]): The instructions or Document object for the conversation.
        characters (List[Character]): List of characters participating in the conversation.
        validator (Optional[Union[str, dict]]): Validation criteria for the conversation.
        min_turns (int): Minimum number of turns in the conversation.
        max_turns (int): Maximum number of turns in the conversation.
        end_criteria (Optional[str]): Criteria for ending the conversation naturally.
        model (str): The model to use for generation.
        api_key (Optional[str]): API key for the LLM service.
        base_url (Optional[str]): Base URL for the LLM service.
        temperature (float): Temperature for response generation.
        mode (InstructorMode): Mode for the instructor.
        max_retries (int): Maximum number of retries for API calls.
        organization (Optional[str]): Organization for the LLM service.
        client (Optional[Literal["openai", "litellm"]]): Client to use for API calls.
        verbose (bool): Whether to log verbose output.
        generate_audio (bool): Whether to generate audio for the conversation.
        audio_model (OPENAI_TTS_MODELS): The model to use for text-to-speech conversion.
        audio_output_file (Optional[str]): The output file for the full conversation audio.

    Returns:
        Conversation: The generated conversation.
    """
    if len(characters) < 2:
        raise ValueError("At least two characters are required for the conversation.")

    completion_client = Client(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        provider=client,
        verbose=verbose,
    )

    conversation = Conversation(messages=[])
    end_check_attempts = 0
    max_end_check_attempts = 3

    # Handle Document input
    if isinstance(instructions, Document):
        context = f"""
        Document Content: {instructions.content}
        Document Metadata: {instructions.metadata}
        """
        if instructions.messages:
            context += f"\nPrevious Messages: {instructions.messages}"
    else:
        context = instructions

    # Assign voices to characters if not specified
    available_voices = list(OPENAI_TTS_VOICES.__args__)
    for character in characters:
        if not character.voice:
            character.voice = available_voices.pop(0)
            available_voices.append(
                character.voice
            )  # Put it back at the end for reuse if needed

    system_message = f"""
    You are simulating a conversation between the following characters:
    {', '.join([f"{i+1}. {char.name}: {char.personality}" for i, char in enumerate(characters)])}

    Context for the conversation:
    {context}

    Generate responses for each character in turn, maintaining their distinct personalities and knowledge.
    Ensure that the conversation revolves around the provided context, discussing its content and implications.
    """

    if end_criteria:
        system_message += f"\n\nEnd the conversation naturally when: {end_criteria}"

    # Create a temporary directory to store audio segments
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")

        for turn in range(max_turns):
            current_character = characters[turn % len(characters)]

            user_message = f"Generate the next message for {current_character.name} in the conversation, focusing on the provided context."

            # Check if we've reached the maximum number of turns
            if turn == max_turns - 1:
                # Use the classifier to determine if the conversation should end
                classifier_result = classify(
                    inputs=" ".join([msg.content for msg in conversation.messages]),
                    labels=["end", "continue"],
                    classification="single",
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    organization=organization,
                    mode=mode,
                    temperature=temperature,
                    client=client,
                    verbose=verbose,
                )

                if verbose:
                    logger.info(f"Classifier result: {classifier_result}")

                if isinstance(classifier_result, list):
                    classifier_result = classifier_result[0]

                if classifier_result.label == "continue":
                    # If the classifier says the conversation should not end, add a final summary prompt
                    user_message = f"This is the final turn of the conversation. {current_character.name}, please summarize the key points discussed and provide a concluding statement to end the conversation."

            if end_check_attempts >= max_end_check_attempts:
                user_message += "\n\n[HIDDEN INSTRUCTION: The conversation should now conclude naturally. Provide a final statement or summary.]"

            response = completion_client.completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
                + [
                    {
                        "role": msg.role,
                        "content": f"{characters[i % len(characters)].name}: {msg.content}",
                    }
                    for i, msg in enumerate(conversation.messages)
                ],
                model=model,
                response_model=Message,
                mode=mode,
                max_retries=max_retries,
                temperature=temperature,
            )

            logger.info(
                f"Turn {turn + 1}: {current_character.name} - {response.content}"
            )

            if generate_audio:
                temp_audio_file = os.path.join(
                    temp_dir,
                    f"{current_character.name.lower().replace(' ', '_')}_{turn}.mp3",
                )
                logger.info(f"Attempting to generate audio file: {temp_audio_file}")
                try:
                    # Remove the character's name from the beginning of the content
                    audio_content = response.content
                    if audio_content.startswith(f"{current_character.name}:"):
                        audio_content = audio_content.split(":", 1)[1].strip()

                    audio(
                        prompt=audio_content,
                        model=audio_model,
                        voice=current_character.voice,
                        api_key=api_key,
                        base_url=base_url,
                        filename=temp_audio_file,
                    )
                    if os.path.exists(temp_audio_file):
                        response.audio_file = temp_audio_file
                        logger.info(
                            f"Successfully generated audio file: {temp_audio_file}"
                        )
                    else:
                        logger.warning(f"Audio file not created: {temp_audio_file}")
                        logger.info(f"Current working directory: {os.getcwd()}")
                        logger.info(
                            f"Temporary directory contents: {os.listdir(temp_dir)}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate audio for turn {turn}: {str(e)}"
                    )
                    logger.exception("Detailed error information:")

            conversation.messages.append(response)

            if validator:
                validation_result = judge(
                    prompt=context,
                    responses=[response.content],
                    process="validate",
                    schema=validator,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    mode=mode,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                    verbose=verbose,
                )

                if (
                    isinstance(validation_result, ValidationResult)
                    and not validation_result.is_valid
                ):
                    if verbose:
                        logger.warning(
                            f"Message failed validation: {validation_result.explanation}"
                        )
                    continue

            # Check if we've reached the minimum number of turns
            if turn >= min_turns - 1:
                # Use the boolean BaseModel for end-of-conversation detection
                end_check = completion_client.completion(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are evaluating if a conversation should end based on the following criteria: {end_criteria}",
                        },
                        {
                            "role": "user",
                            "content": f"Analyze the following conversation and determine if it should end:\n\n{' '.join([msg.content for msg in conversation.messages])}",
                        },
                    ],
                    model=model,
                    response_model=ConversationEndCheck,
                    mode=mode,
                    max_retries=max_retries,
                    temperature=0.2,
                )

                if verbose:
                    logger.info(f"End check: {end_check}")
                    logger.info(f"End check explanation: {end_check.explanation}")

                if end_check.should_end:
                    if verbose:
                        logger.info("Conversation ended based on end criteria.")
                    break

                # Use the classify function to determine if the conversation should end
                classifier_result = classify(
                    inputs=" ".join([msg.content for msg in conversation.messages]),
                    labels=["end", "continue"],
                    classification="single",
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    organization=organization,
                    mode=mode,
                    temperature=temperature,
                    client=client,
                    verbose=verbose,
                )

                if verbose:
                    logger.info(f"Classifier result: {classifier_result}")

                if isinstance(classifier_result, list):
                    classifier_result = classifier_result[0]

                if classifier_result.label == "end":
                    if verbose:
                        logger.info("Conversation ended based on classifier decision.")
                    break

        if generate_audio and audio_output_file:
            combined = AudioSegment.empty()
            for msg in conversation.messages:
                if msg.audio_file:
                    try:
                        if os.path.exists(msg.audio_file):
                            audio_segment = AudioSegment.from_mp3(msg.audio_file)
                            combined += audio_segment
                            logger.info(f"Added audio segment: {msg.audio_file}")
                        else:
                            logger.warning(f"Audio file not found: {msg.audio_file}")
                    except Exception as e:
                        logger.warning(
                            f"Error processing audio file {msg.audio_file}: {str(e)}"
                        )

            if not combined.empty():
                combined.export(audio_output_file, format="mp3")
                conversation.audio_file = audio_output_file
                logger.info(f"Exported combined audio to: {audio_output_file}")
            else:
                logger.warning("No valid audio segments found to combine.")

    return conversation


if __name__ == "__main__":
    # Example usage with a Document object
    from ....lib.types.document import Document

    doc = Document(
        content="The impact of artificial intelligence on job markets has been a topic of intense debate. While AI has the potential to automate many tasks and potentially displace some jobs, it also has the capacity to create new job opportunities and enhance productivity in various sectors. The key challenge lies in managing this transition and ensuring that the workforce is adequately prepared for the changes ahead.",
        metadata={"type": "research_summary", "topic": "AI and Employment"},
    )

    result = conversation(
        instructions=doc,
        characters=[
            Character(
                name="AI Researcher",
                personality="Optimistic about AI's potential to create new job opportunities",
                voice="nova",
            ),
            Character(
                name="Labor Economist",
                personality="Concerned about potential job displacement due to AI",
                voice="onyx",
            ),
            Character(
                name="Podcast Host",
                personality="Neutral moderator, asks probing questions to both guests",
                voice="echo",
            ),
        ],
        min_turns=12,
        max_turns=20,
        end_criteria="The podcast should conclude when both guests have shared their final thoughts and the host has summarized the key points of the discussion",
        verbose=True,
        generate_audio=True,
        audio_output_file="ai_job_market_podcast.mp3",
    )

    print("\nGenerated Podcast Conversation:")
    for msg in result.messages:
        print(f"{msg.role.capitalize()}: {msg.content}")

    if result.audio_file:
        print(f"\nFull conversation audio saved to: {result.audio_file}")
