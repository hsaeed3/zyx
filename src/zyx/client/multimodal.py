from typing import Literal, Optional, Any, Union

__all__ = ["image", "speak", "transcribe"]

ModelType = Literal[
    "dall-e-2",
    "dall-e-3",
    "flux-dev",
    "flux-realism",
    "flux-schnell",
    "flux-pro",
    "flux-lora",
    "flux-general",
    "aura",
    "sd-v3",
    "fooocus",
]

PREDEFINED_APPS: Literal[
    "fal-ai/flux/dev",
    "fal-ai/flux-realism",
    "fal-ai/flux/schnell",
    "fal-ai/flux-pro",
    "fal-ai/flux-lora",
    "fal-ai/flux-general",
    "fal-ai/aura-flow",
    "fal-al/lora",
    "fal-ai/stable-diffusion-v3-medium",
    "fal-ai/fooocus",
] = None


def _get_model_config(model: ModelType):
    dall_e_models = ["dall-e-2", "dall-e-3"]
    fal_models = {
        "flux-dev": "fal-ai/flux/dev",
        "flux-realism": "fal-ai/flux-realism",
        "flux-schnell": "fal-ai/flux/schnell",
        "flux-pro": "fal-ai/flux-pro",
        "flux-lora": "fal-ai/flux-lora",
        "flux-general": "fal-ai/flux-general",
        "aura": "fal-ai/aura-flow",
        "sd-v3": "fal-ai/stable-diffusion-v3-medium",
        "fooocus": "fal-ai/fooocus",
    }

    if model in dall_e_models:
        return {"provider": "openai", "model": model}
    elif model in fal_models:
        return {"provider": "fal", "application": fal_models[model]}
    else:
        raise ValueError(f"Unsupported model: {model}")


def image(
    prompt: str,
    model: ModelType = "dall-e-3",
    api_key: Optional[str] = None,
    image_size: Optional[str] = "landscape_4_3",
    num_inference_steps: Optional[int] = 26,
    guidance_scale: Optional[float] = 3.5,
    enable_safety_checker: Optional[bool] = False,
    size: Optional[str] = "1024x1024",
    quality: Optional[str] = "standard",
    n: Optional[int] = 1,
    display: Optional[bool] = False,
    optimize_prompt : Optional[bool] = False,
    optimize_prompt_model : Optional[str] = "openai/gpt-4o-mini",
) -> Union[str, Any]:
    """Generates an image using either the FAL_AI API or OpenAI. With an
    optional display function to show the image in a notebook.

    Parameters:
        prompt: str,
        model: ModelType = "dall-e-3",
        api_key: Optional[str] = None,
        image_size: Optional[str] = "landscape_4_3",
        num_inference_steps: Optional[int] = 26,
        guidance_scale: Optional[float] = 3.5,
        enable_safety_checker: Optional[bool] = False,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = "standard",
        n: Optional[int] = 1,
        display: Optional[bool] = False,

    Returns:
        str or Any: The generated image or an error message.
    """
    model_config = _get_model_config(model)

    if model_config["provider"] == "openai":
        from openai import OpenAI

        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            return e
        try:
            response = client.images.generate(
                model=model_config["model"],
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
        except Exception as e:
            return e
        if display:
            try:
                from IPython.display import display, Image
            except ImportError:
                from .. import logger

                raise logger.critical(
                    "The display function requires IPython, which is not included in the base 'zyx' package. Please install it with `pip install ipython`."
                )

            url = response.data[0].url
            display(Image(url=url))
        return response

    elif model_config["provider"] == "fal":
        import fal_client

        if optimize_prompt:
            from ..client.main import completion
            from ..core.main import BaseModel
            
            class OptimizedPrompt(BaseModel):
                prompt : str
                
            optimized_prompt = completion(
                messages = [
                {"role" : "system", "content" : f"The original prompt is : [ {prompt} ]. Refactor it to include a lot more description."},
                {"role" : "user", "content" : "Optimize this prompt for use in image generation. DO NOT INCLUDE ANYTHING ELSE."}
                ],
                model = optimize_prompt_model,
                response_model=OptimizedPrompt
            )
            
            prompt = optimized_prompt.prompt

        try:
            handler = fal_client.submit(
                application=model_config["application"],
                arguments={
                    "prompt": prompt,
                    "image_size": image_size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "enable_safety_checker": enable_safety_checker,
                    "num_images": n,
                },
            )
            result = handler.get()
            if display:
                try:
                    from IPython.display import display, Image
                except ImportError:
                    from .. import logger

                    raise logger.critical(
                        "The display function requires IPython, which is not included in the base 'zyx' package. Please install it with `pip install ipython`."
                    )
                url = result["images"][0]["url"]
                display(Image(url=url))
        except Exception as e:
            result = e
        return result


OPENAI_TTS_MODELS = Literal["tts-1", "tts-1-hd"]
OPENAI_TTS_VOICES = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def speak(
    prompt: str,
    model: OPENAI_TTS_MODELS = "tts-1",
    voice: OPENAI_TTS_VOICES = "alloy",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    filename: Optional[str] = None,
    play: bool = False,
):
    """Generates an audio file from text, through the openai API.

    Parameters:
        prompt: str,
        model: OPENAI_TTS_MODELS = "tts-1",
        voice: OPENAI_TTS_VOICES = "alloy",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        filename: Optional[str] = None,
        play: bool = False,

    Returns:
        str or Any: The generated audio file or an error message.
    """
    from openai import OpenAI
    import io

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        from .. import logger

        raise logger.critical(
            "The [italic]speak[/italic] function requires sounddevice and soundfile, which are not included in the base 'zyx' package. Please install them with [bold]`pip install sounddevice soundfile`[/bold]."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.audio.speech.create(input=prompt, model=model, voice=voice)
        audio_data = response.read()

        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)

        if filename:
            file_endings = [".wav", ".mp3", ".m4a"]
            if not filename.endswith(tuple(file_endings)):
                raise ValueError(
                    f"Filename must end with one of the following: {', '.join(file_endings)}"
                )

            sf.write(filename, audio_array, sample_rate)

        if play:
            try:
                from IPython.display import Audio
            except ImportError:
                from .. import logger

                raise logger.critical(
                    "The [italic]play[/italic] function requires IPython, which is not included in the base 'zyx' package. Please install it with [bold]`pip install ipython`[/bold]."
                )

            # Play audio using sounddevice
            sd.play(audio_array, sample_rate)
            sd.wait()

            # For Jupyter notebook, also return IPython audio widget
            return Audio(audio_array, rate=sample_rate, autoplay=True)
        else:
            return audio_array, sample_rate

    except Exception as e:
        return str(e)


def transcribe(
    model: str = "whisper-1",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    file: Optional[str] = None,
    record: bool = False,
    duration: int = 5,
):
    """Transcribes an audio file into text, through the openai API.

    Parameters:
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        file: Optional[str] = None,
        record: bool = False,
        duration: int = 5,

    Returns:
        str or Any: The transcribed text or an error message.
    """
    from openai import OpenAI

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        from .. import logger

        raise logger.critical(
            "The [italic]speak[/italic] function requires sounddevice and soundfile, which are not included in the base 'zyx' package. Please install them with [bold]`pip install sounddevice soundfile`[/bold]."
        )

    import io

    client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)

    if record:
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
        sd.wait()
        print("Recording finished.")

        with io.BytesIO() as buffer:
            sf.write(buffer, audio_data, 44100, format="wav")
            buffer.seek(0)
            audio_file = buffer
    elif file:
        if not file.endswith((".mp3", ".wav", ".m4a")):
            raise ValueError("File must be a .mp3, .wav, or .m4a file")
        audio_file = open(file, "rb")
    else:
        raise ValueError(
            "Either 'file' must be provided or 'record' must be set to True"
        )

    try:
        transcription = client.audio.transcriptions.create(
            model=model, file=audio_file, response_format="text"
        )
        return transcription
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print(
        image(
            model="flux-dev",
            prompt="A beautiful landscape painting of a sunset over the ocean.",
            n=1,
        )
    )
