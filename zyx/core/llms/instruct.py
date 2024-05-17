# >>>>>>>>>>>>>>>>>>>>>>>>>>>
# zyx is open source
# use it however you want :)
#
# 2024 Hammad Saeed
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<

import anthropic
import instructor
from loguru import logger
from openai import OpenAI
from pydantic import Field, BaseModel as PydanticBaseModel
from typing import Optional, Union
from zyx.core.llms.functions.completion import completion

COSTAR_PROMPT_INSTRUCTION = """
The Co-Star prompting framework is an incredibly advanced and powerful tool that allows you to generate highly specific and contextually relevant prompts for your LLMs.
The structure of a Co-Star Prompt is as follows:

1. Context: Provides background information that helps the LLM understand the specific scenario.
2. Objective: Defines the goal of the response.
3. Style: Defines the style of the response.
4. Tone: Defines the tone of the response.
5. Audience: Defines the audience the response is intended for.
6. Response Format: Defines the format of the response.

Create a very well defined co-star prompt to reach the best possible results for this specific role = {role}.
"""

COSTAR_PROMPT_TEMPLATE = """
## CONTEXT ##
{context}

## OBJECTIVE ##
{objective}

## STYLE ##
{style}

## TONE ##
{tone}

## AUDIENCE ##
{audience}

## RESPONSE FORMAT ##
{response_format}
"""

class CoStarSystemPrompt(PydanticBaseModel):
    context : str = Field(str, description="Provides background information helps the LLM understand the specific scenario.")
    objective : str = Field(str, description="Defines the goal of the response.")
    style : str = Field(str, description="Defines the style of the response.")
    tone : str = Field(str, description="Defines the tone of the response.")
    audience : str = Field(str, description="Defines the audience the response is intended for.")
    response_format : str = Field(str, description="Defines the format of the response.")

class Instruct:
    class BaseModel(PydanticBaseModel):
        # This nested class acts as a proxy to Pydantic's BaseModel, making it accessible as Instruct.BaseModel
        pass
    def __init__ (self,
                  role : str,
                  model : Optional[str] = "openai/gpt-3.5-turbo",
                  api_key : Optional[str] = None,
                  ):
        """
        A class for creating a 'tuned' Instructed response.

        Args:
            model (Optional[str]): The model to use for completion. Defaults to "openai/gpt-3.5-turbo".
        """
        if not role:
            raise logger.error("Role is required to initialize the Instruct class!")

        if not model:
            raise logger.error("Model is required to initialize the Instruct class!")
        
        if not model.startswith("anthropic/") and not model.startswith("openai/"):
            raise logger.error("The instruct class uses the LiteLLM model syntax. The model must start with 'anthropic/' or 'openai/'.")
        
        self.role = role
        self.model = model
        self.litellm_model = model
        self.api_key = api_key

        self.instructor = self._init_model()
        self.system_prompt = self._build_costar_prompt(role = self.role)

    def _init_model(self) -> instructor.Instructor:
        # Check if model is anthropic or openai
        if self.model.startswith("anthropic/"):
            self.model = self.model[10:]
            return instructor.from_anthropic(anthropic.Anthropic(api_key = self.api_key))
        elif self.model.startswith("openai/"):
            self.model = self.model[7:]
            return instructor.from_openai(OpenAI(api_key = self.api_key))
        
    def _build_costar_prompt(self, role : str) -> str:
        # Preconditions:
        if not self.instructor:
            raise logger.error("Instructor is required to build a Co-Star Prompt!")
        query_prompt = COSTAR_PROMPT_INSTRUCTION.format(role = role)

        messages = [{"role": "user", "content": query_prompt}]
        try:
            generated_response = self.instructor.chat.completions.create(
                response_model = CoStarSystemPrompt,
                model = self.model,
                messages = messages,
                max_retries = 3,
            )
        except Exception as e:
            raise logger.error(f"Error while generating the Co-Star Prompt: {e}")
        try:
            co_star_prompt = COSTAR_PROMPT_TEMPLATE.format(
                context = generated_response.context,
                objective = generated_response.objective,
                style = generated_response.style,
                tone = generated_response.tone,
                audience = generated_response.audience,
                response_format = generated_response.response_format,
            )
        except Exception as e:
            raise logger.error(f"Error while generating the Co-Star Prompt: {e}")
        return co_star_prompt
    
    def instruct(self, prompt : str = None, model : Optional[Union[list, str]] = None, api_key : Optional[str] = None, response_model : Optional[BaseModel] = None, *args, **kwargs) -> str:
        if not self.system_prompt:
            raise logger.error("System Prompt is required to instruct the model!")

        if model is not None:
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}]
            
            if not api_key:
                api_key = self.api_key if self.api_key else None
            
            try:
                response = completion(
                    messages = messages,
                    model = model,
                    api_key = api_key,
                    *args, **kwargs
                )
                if response_model is not None:
                    instructed_regeneration = self.instructor.chat.completions.create(
                        messages = [
                            {"role": "system", "content": "You are an instructor, that does not regenerate anything; but rather appends inputs. You are not legally allowed to ever change the text that you are given."},
                            {"role": "user", "content": response}
                        ],
                        model = self.model,
                        response_model = response_model,
                        max_retries = 3,
                        *args, **kwargs
                    )
                    response = instructed_regeneration
                return response

            except Exception as e:
                raise logger.error(f"Error while instructing the model: {e}")
        else:
            if response_model is not None:
                messages = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}]
                
                if not api_key:
                    api_key = self.api_key if self.api_key else None
                
                try:
                    response = self.instructor.chat.completions.create(
                        messages = messages,
                        model = self.model,
                        response_model = response_model,
                        *args, **kwargs
                    )
                    return response
                except Exception as e:
                    raise logger.error(f"Error while instructing the model: {e}")
            else:
                messages = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}]
                
                if not api_key:
                    api_key = self.api_key if self.api_key else None

                model = self.litellm_model
                
                try:
                    response = completion(
                        messages = messages,
                        model = model,
                        api_key = api_key,
                        *args, **kwargs
                    )
                    return response
                except Exception as e:
                    raise logger.error(f"Error while instructing the model: {e}")
            

if __name__ == "__main__":
    instruct = Instruct("A fastapi expert")

    response = instruct.instruct("Write me a python script creating an inference endpoint")
    print(response)

    class Response(Instruct.BaseModel):
        explanation : str
        code : str

    response = instruct.instruct("Write me a python script creating an inference endpoint",
                                 model = "ollama/llama3",
                                 response_model = Response)
    print(response)