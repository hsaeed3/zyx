"""
### zyx._api_resource

API client resource module used for chat completions, multi-modal generations and other
related tasks. This module acts as a manager, providing a unified interface between the
`LiteLLM` & `OpenAI` API clients, as well as a resource for patching the client with
the `Instructor` library for structured outputs.
"""

from __future__ import annotations

from typing import Optional, Type, Union, Mapping
import httpx
import time
import instructor
from openai import OpenAI, AsyncOpenAI

from .types.client.client_config import ClientConfig
from .types.client.client_provider_recommendation import ClientProviderRecommendation
from .types.client.client_provider import ClientProvider
from .types.instructor import InstructorMode
from zyx import logging


# Type Hint for LiteLLM's Client class
LiteLLM = Type["LiteLLM"]


# ===================================================================
# [Singletons]
# ===================================================================


# litellm resource singleton
_litellm_resource: Optional[LiteLLMResource] = None
"""
The singleton helper instance for the `LiteLLM` library.
"""


# ===================================================================
# [LiteLLM Resource]
# ===================================================================


class LiteLLMResource:
    """
    Singleton resource helper for the `LiteLLM` library.
    """

    _instance: Optional[LiteLLMResource] = None
    litellm = None

    def __new__(cls) -> LiteLLMResource:
        if cls._instance is None:
            start_time = time.time()
            import litellm

            end_time = time.time()
            if logging.get_verbosity_level() == 1:
                logging.verbose_print(
                    f"Performed first time on-run import of [bold plum3]LiteLLM[/bold plum3] in [italic]{end_time - start_time:.2f} seconds.[/italic]"
                )
                logging.logger.debug(f"imported litellm once in {end_time - start_time:.2f} seconds")
            cls.litellm = litellm
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_litellm():
        """
        Returns the singleton `LiteLLM` client.
        """
        global _litellm_resource
        if not _litellm_resource:
            _litellm_resource = LiteLLMResource()
        return _litellm_resource.litellm


# ===================================================================
# API Resource
# ===================================================================


class APIResource:
    """
    Singleton resource manager for the `OpenAI` & `Instructor` API clients. Provides an integration
    to patch with the `Instructor` library for structured outputs.
    """

    provider: Optional[ClientProvider] = None
    """The provider this manager is using."""

    # [Attributes]
    _client: Union[OpenAI, LiteLLM] = None
    """The base client to be used internally by the manager.
    
    Can be one of:
    - openai.OpenAI
    - litellm.LiteLLM
    - litellm.completion
    """

    client = None
    """The initialized client"""

    patch: Optional[Union[instructor.Instructor, instructor.AsyncInstructor]] = None
    """The currently active patch instance from the `Instructor` library."""

    # ===================================================================
    # [Main Methods]
    # ===================================================================

    def __init__(
        self,
        provider: Optional[ClientProvider] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        project: Optional[str] = None,
        default_query: Optional[Mapping[str, object]] = None,
        websocket_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        _strict_response_validation: Optional[bool] = False,
        config: Optional[ClientConfig] = None,
        client: Optional[Union[OpenAI, LiteLLM]] = None,
        verbose: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        # set default attributes
        self.provider = None
        self.client = None
        self.patch = None

        # set verbosity level
        if verbose:
            logging.set_verbose(True)
        if debug:
            logging.set_debug(True)

        if client is not None:
            # simple check for openai or litellm for checking for chat attr
            if isinstance(client, OpenAI) or hasattr(client, "chat"):
                # set client to self
                self.client = client
                # set provider
                if isinstance(client, OpenAI):
                    self.provider = "openai"
                else:
                    self.provider = "litellm"
                logging.logger.debug(f"recieved & set client instance on init: {client}")
                logging.logger.debug(f"predicted client provider: {self.provider}")

        else:
            config = APIResource.create_config(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                project=project,
                default_query=default_query,
                websocket_base_url=websocket_base_url,
                http_client=http_client,
                _strict_response_validation=_strict_response_validation,
                config=config,
            )
            non_none_config_values = {k: v for k, v in config.dump_for_openai().items() if v is not None}
            if non_none_config_values:
                logging.logger.debug(f"recieved config values on init: {non_none_config_values}")
            else:
                logging.logger.debug(f"no config values recieved on init")

            if (
                any(
                    [
                        config.api_key,
                        config.base_url,
                        config.organization,
                        config.timeout,
                        config.max_retries,
                        config.default_headers,
                        config.project,
                        config.default_query,
                        config.websocket_base_url,
                        config.http_client,
                        config._strict_response_validation,
                    ]
                )
                or provider is not None
            ):
                if provider is not None:
                    logging.logger.debug(f"recieved client provider on init: {provider}")

                # !! 
                # load client
                self.load_client(provider=provider, config=config)

            else:
                logging.logger.debug(f"no config or provider recieved on initialization.. skipping client creation")

    def create_client(
        self,
        provider: Optional[ClientProvider] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        project: Optional[str] = None,
        default_query: Optional[Mapping[str, object]] = None,
        websocket_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        _strict_response_validation: Optional[bool] = False,
        config: Optional[ClientConfig] = None,
    ) -> None:
        """
        Creates a new client instance.
        """

        config = APIResource.create_config(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            project=project,
            default_query=default_query,
            websocket_base_url=websocket_base_url,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            config=config,
        )

        if provider not in ["openai", "litellm"]:
            raise ValueError(f"Invalid provider: {provider}")

        start_time = time.time()
        if provider == "openai":
            self.client = APIResource.create_openai_client(config=config)
        else:
            self.client = APIResource.create_litellm_client(config=config)
        self.provider = provider
        end_time = time.time()

        if logging.get_verbosity_level() == 1:
            logging.verbose_print(
                f"Initialized a new base client using [bold plum3]{'OpenAI' if provider == 'openai' else 'LiteLLM'}[/bold plum3] in [italic]{end_time - start_time:.2f} seconds.[/italic]"
            )
        logging.logger.debug(
            f"initialized new base client using {provider} in {end_time - start_time:.2f} seconds using params: {config.dump_for_openai()}"
        )

    def load_client(
        self,
        provider: Optional[ClientProvider] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        project: Optional[str] = None,
        default_query: Optional[Mapping[str, object]] = None,
        websocket_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        _strict_response_validation: Optional[bool] = False,
        config: Optional[ClientConfig] = None,
    ) -> None:
        """
        Loads or updates the current client instance.
        """

        # Determine if a reload is necessary
        needs_reload = not self.client or provider != self.provider

        # Set the provider if not provided
        provider = provider or self.provider or "openai"

        # If the provider has changed or the client is not loaded, reload the client
        if needs_reload:
            self.create_client(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                project=project,
                default_query=default_query,
                websocket_base_url=websocket_base_url,
                http_client=http_client,
                _strict_response_validation=_strict_response_validation,
                config=config,
            )
            self.provider = provider
            return

        # Update existing client attributes if the provider is the same
        config_attrs = [
            "api_key",
            "base_url",
            "organization",
            "timeout",
            "max_retries",
            "default_headers",
            "project",
            "default_query",
            "websocket_base_url",
            "http_client",
            "_strict_response_validation",
        ]

        for attr in config_attrs:
            config_value = getattr(config, attr, None) if config else None
            current_value = getattr(self.client, attr, None)
            new_value = config_value if config_value is not None else locals()[attr]

            if new_value is not None and new_value != current_value:
                setattr(self.client, attr, new_value)

    def load_patch(
        self,
        mode: Optional[InstructorMode] = "tool_call",
    ) -> None:
        """
        Updates the current client patch with Instructor & given mode.
        """

        if self.client is None:
            logging.warn("No OpenAI or LiteLLM client is loaded yet, cannot patch None client!")
            return

        if self.patch:
            # ensure hooks are set
            APIResource.set_instructor_hooks(self.patch)
            return

        if self.provider == "openai":
            self.patch = APIResource.create_instructor_patch_for_openai(self.client, mode=mode)
            APIResource.set_instructor_hooks(self.patch)

        else:
            self.patch = APIResource.create_instructor_patch_for_litellm(self.client, mode=mode)
            APIResource.set_instructor_hooks(self.patch)

        if logging.get_verbosity_level() == 1:
            logging.verbose_print(
                f"Successfully loaded instructor patch for [bold plum3]{'OpenAI' if self.provider == 'openai' else 'LiteLLM'}[/bold plum3] client, using Instructor mode: [bold green]{mode}[/bold green]"
            )
        return

    # ===================================================================
    # [Static Methods]
    # ===================================================================

    @staticmethod
    def recommend_provider(
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ClientProviderRecommendation:
        """
        Recommends the best provider for the given model.
        """
        try:
            recommendation = ClientProviderRecommendation.recommend(model, base_url, api_key)
            logging.logger.debug(
                f"successfully recommended provider : {recommendation.provider} for model: {model}, base_url: {base_url}, api_key: {api_key}"
            )
            return recommendation
        except Exception as e:
            logging.logger.error(
                f"Failed to recommend provider for model: {model}, base_url: {base_url}, api_key: {api_key}"
            )
            raise e

    @staticmethod
    def create_config(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        project: Optional[str] = None,
        default_query: Optional[Mapping[str, object]] = None,
        websocket_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        _strict_response_validation: Optional[bool] = False,
        config: Optional[ClientConfig] = None,
    ) -> ClientConfig:
        """
        Creates a new `ClientConfig` instance.
        """
        config_params = {
            "api_key": api_key,
            "base_url": base_url,
            "organization": organization,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "project": project,
            "default_query": default_query,
            "websocket_base_url": websocket_base_url,
            "http_client": http_client,
            "_strict_response_validation": _strict_response_validation,
        }

        if config:
            existing_params = config.model_dump()
            existing_params.update({k: v for k, v in config_params.items() if v is not None})
            return ClientConfig(**existing_params)

        return ClientConfig(**{k: v for k, v in config_params.items() if v is not None})

    @staticmethod
    def create_openai_client(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        project: Optional[str] = None,
        default_query: Optional[Mapping[str, object]] = None,
        websocket_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        _strict_response_validation: Optional[bool] = False,
        config: Optional[ClientConfig] = None,
    ) -> OpenAI:
        """
        Creates a new instance of the `OpenAI` client.
        """
        config = APIResource.create_config(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            project=project,
            default_query=default_query,
            websocket_base_url=websocket_base_url,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            config=config,
        )
        try:
            client_params = {k: v for k, v in config.dump_for_openai().items() if v is not None}
            client = OpenAI(**client_params)
            return client
        except Exception as e:
            logging.logger.error(f"Failed to create `OpenAI` client: {e}")
            raise e

    @staticmethod
    def create_litellm_client(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout, None]] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        config: Optional[ClientConfig] = None,
    ) -> LiteLLM:
        """
        Creates a new instance of the `LiteLLM` client.
        """
        config = APIResource.create_config(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            config=config,
        )
        try:
            client_params = {k: v for k, v in config.dump_for_litellm().items() if v is not None}
            client = LiteLLMResource.get_litellm().LiteLLM(**client_params)
            return client
        except Exception as e:
            logging.logger.error(f"Failed to create `LiteLLM` client: {e}")
            raise e

    @staticmethod
    def create_instructor_patch_for_openai(
        client: OpenAI | AsyncOpenAI,
        mode: str = "tool_call",
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        """
        Creates a new instructor patch for the given client.
        """
        try:
            patch = instructor.from_openai(client, mode=instructor.Mode(mode))
            logging.logger.debug(
                f"successfully initialized new instructor patch for `OpenAI` client using mode : {mode}"
            )
            return patch
        except Exception as e:
            logging.logger.error(f"Failed to create instructor patch for `OpenAI` client: {e}")
            raise e

    @staticmethod
    def create_instructor_patch_for_litellm(
        client: LiteLLM,
        mode: str = "tool_call",
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        """
        Creates a new instructor patch for the given client.
        """
        try:
            patch = instructor.from_litellm(client, mode=instructor.Mode(mode))
            logging.logger.debug(
                f"successfully initialized new instructor patch for `LiteLLM` client using mode : {mode}"
            )
            return patch
        except Exception as e:
            logging.logger.error(f"Failed to create instructor patch for `LiteLLM` client: {e}")
            raise e

    @staticmethod
    def get_instructor_raw_response(response):
        """
        Returns the raw response created by instructor chat completions.
        """
        logging.logger.debug(f"completion passed through instructor response hook, raw response: {response}")
        return response

    @staticmethod
    def set_instructor_hooks(
        client: Union[instructor.Instructor, instructor.AsyncInstructor],
    ) -> None:
        """
        Sets the hooks for the instructor client.
        """
        client.on(
            hook_name="completion:response",
            handler=APIResource.get_instructor_raw_response,
        )
        logging.logger.debug("set instructor hooks for: raw response collection")




