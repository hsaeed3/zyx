"""zyx.models.embeddings.model"""

from __future__ import annotations

import logging
import asyncio
from collections.abc import Iterable
from typing import (
    TypeVar,
)

from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)

from ..providers import (
    ModelProvider,
    ModelProviderName,
    ModelProviderRegistry,
)
from ..clients import ModelClient
from ..clients.openai import OpenAIModelClient
from ..clients.litellm import LiteLLMModelClient
from .types import (
    EmbeddingModelName,
    EmbeddingEncodingFormat,
)

__all__ = [
    "EmbeddingModel",
]


_logger = logging.getLogger(__name__)


T = TypeVar("T")


class EmbeddingModel:
    """A simple, unified interface around embedding model provider clients,
    (OpenAI, LiteLLM) for generating embeddings."""

    def __init__(
        self,
        model: EmbeddingModelName | str = "openai/text-embedding-3-small",
        *,
        provider: ModelProviderName | ModelProvider | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        user: str | None = None,
    ) -> None:
        """Initialize the language model with a model name, and optional provider
        name / custom base URL and API key, along with an optional set of
        settings & tools to use as default for this model.

        NOTE: A `LanguageModel` instance does not run tools.
        """
        self._model = model
        self._client: ModelClient[T] | None = None
        self._provider: ModelProvider | None = None

        if not dimensions:
            # try auto infer, else we leave as None
            if self._model.startswith("openai/") or self._model.startswith(
                "text-embedding-"
            ):
                self._dimensions = 1536

            else:
                self._dimensions = None
        else:
            self._dimensions = dimensions

        self._encoding_format = encoding_format if encoding_format else None
        self._user = user if user else None

        if base_url:
            assert not provider, "Cannot provide base_url and a provider"

            # try to get custom provider if exists, all custom providers
            # use an OpenAI client
            if ModelProviderRegistry().get(f"custom:{base_url}"):
                self._provider = ModelProviderRegistry().get(f"custom:{base_url}")
            else:
                self._provider = ModelProviderRegistry().register_custom(
                    base_url, api_key
                )

        if provider:
            if isinstance(provider, str):
                self._provider = ModelProviderRegistry().get(provider)

                if not self._provider:
                    raise ValueError(f"Unknown provider: {provider}")

            elif isinstance(provider, ModelProvider):
                self._provider = provider
            else:
                raise ValueError(f"Invalid provider: {provider}")

        else:
            self._provider = ModelProviderRegistry().infer_from_model_name(
                model=self._model,
                kind="embedding_model",
            )

            if not self._provider:
                # this will fallback to the LiteLLM client
                self._provider = ModelProvider(name="unknown")

    def __str__(self):
        from ..._internal._beautification import _pretty_print_model

        return _pretty_print_model(self, "embedding_model")

    def __rich__(self):
        from ..._internal._beautification import _rich_pretty_print_model

        return _rich_pretty_print_model(self, "embedding_model")

    @property
    def model(self) -> EmbeddingModelName | str:
        """Get the model name for this embedding model."""
        return self._model

    @property
    def dimensions(self) -> int | None:
        """Get the dimensions for this embedding model."""
        return self._dimensions

    @property
    def encoding_format(self) -> EmbeddingEncodingFormat | None:
        """Get the encoding format for this embedding model."""
        return self._encoding_format

    @property
    def user(self) -> str | None:
        """Get the user for this embedding model."""
        return self._user

    def get_client(self) -> ModelClient[T]:
        """Get the inferred / associated model client for this language model.

        This requires the `_provider` attribute to be set as a ModelProvider
        object."""

        if not self._provider:
            raise ValueError("Cannot get client without a provider")

        if not self._client:
            if "openai" in self._provider.supported_clients:
                self._client = OpenAIModelClient(
                    base_url=self._provider.base_url,
                    api_key=self._provider.get_api_key(),
                )
            else:
                self._client = LiteLLMModelClient(
                    base_url=self._provider.base_url,
                    api_key=self._provider.get_api_key(),
                )

        return self._client

    async def arun(
        self,
        input: str | Iterable[str],
        *,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        user: str | None = None,
    ) -> EmbeddingModelResponse:
        """Asyncronously run an embedding model with a given input, and optional dimensions, encoding format, and user.

        Parameters
        ----------
        input : str | Iterable[str]
            The input to generate embeddings for.
        dimensions : int | None
            The dimensions to generate embeddings for.
        encoding_format : EmbeddingEncodingFormat | None
        user : str | None
        """
        if not self._provider:
            raise ValueError("Cannot run model without a provider")

        params = {
            "model": self._provider.clean_model_name(self._model),
            "input": input,
        }

        # Use provided dimensions or fall back to instance dimensions
        final_dimensions = dimensions if dimensions is not None else self._dimensions
        if final_dimensions is not None:
            params["dimensions"] = final_dimensions

        # Use provided encoding_format or fall back to instance encoding_format
        final_encoding_format = (
            encoding_format if encoding_format is not None else self._encoding_format
        )
        if final_encoding_format is not None:
            params["encoding_format"] = final_encoding_format

        # Use provided user or fall back to instance user
        final_user = user if user is not None else self._user
        if final_user is not None:
            params["user"] = final_user

        return await self.get_client().embedding(**params)

    def run(
        self,
        input: str | Iterable[str],
        *,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        user: str | None = None,
    ) -> EmbeddingModelResponse:
        """Run an embedding model with a given input, and optional dimensions, encoding format, and user."""
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.arun(
                        input=input,
                        dimensions=dimensions,
                        encoding_format=encoding_format,
                        user=user,
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(
                self.arun(
                    input=input,
                    dimensions=dimensions,
                    encoding_format=encoding_format,
                    user=user,
                )
            )
