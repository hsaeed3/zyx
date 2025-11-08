"""zyx.models.embeddings.model"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)

from ..definition import ModelDefinition
from ..providers import ModelProvider, ModelProviderName
from .types import EmbeddingEncodingFormat, EmbeddingModelName, EmbeddingModelSettings

__all__ = [
    "EmbeddingModel",
    "arun_embed",
    "run_embed",
    "embedder",
]


_logger = logging.getLogger(__name__)


class EmbeddingModel(ModelDefinition):
    """A simple, unified interface around embedding model provider clients,
    (OpenAI, LiteLLM) for generating embeddings."""

    @property
    def kind(self) -> str:
        return "embedding_model"

    @property
    def model(self) -> EmbeddingModelName | str:
        """Get the model name for this embedding model."""
        return self._model

    @property
    def settings(self) -> EmbeddingModelSettings:
        """Get the settings for this embedding model."""
        return self._settings

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
        if not dimensions:
            # try auto infer, else we leave as None
            if model.startswith("openai/") or model.startswith("text-embedding-"):
                dimensions = 1536

            else:
                dimensions = None
        else:
            dimensions = dimensions

        settings = EmbeddingModelSettings(
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        )

        super().__init__(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            settings=settings,
        )

    async def arun(
        self,
        input: str | Iterable[str],
        *,
        dimensions: int | None = None,
        encoding_format: EmbeddingEncodingFormat | None = None,
        user: str | None = None,
    ) -> EmbeddingModelResponse:
        """Asynchronously run an embedding model with a given input.

        Parameters
        ----------
        input : str | Iterable[str]
            The input to generate embeddings for.
        dimensions : int | None
            The dimensions to generate embeddings for.
        encoding_format : EmbeddingEncodingFormat | None
        user : str | None
            The user to generate embeddings for.

        Returns
        -------
        EmbeddingModelResponse
            The response from the embedding model.
        """
        if not self._provider:
            raise ValueError("Cannot run model without a provider")

        params = {
            "model": self._provider.clean_model_name(self._model),
            "input": input,
        }

        # Use provided dimensions or fall back to instance dimensions
        final_dimensions = (
            dimensions if dimensions is not None else self.settings.dimensions
        )
        if final_dimensions is not None:
            params["dimensions"] = final_dimensions

        # Use provided encoding_format or fall back to instance encoding_format
        final_encoding_format = (
            encoding_format
            if encoding_format is not None
            else self.settings.encoding_format
        )
        if final_encoding_format is not None:
            params["encoding_format"] = final_encoding_format

        # Use provided user or fall back to instance user
        final_user = user if user is not None else self.settings.user
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
        """Run an embedding model with a given input.

        Parameters
        ----------
        input : str | Iterable[str]
            The input to generate embeddings for.
        dimensions : int | None
            The dimensions to generate embeddings for.
        encoding_format : EmbeddingEncodingFormat | None
        user : str | None
            The user to generate embeddings for.

        Returns
        -------
        EmbeddingModelResponse
            The response from the embedding model.
        """
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


async def arun_embed(
    input: str | Iterable[str],
    model: EmbeddingModelName | str = "openai/text-embedding-3-small",
    *,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
    encoding_format: EmbeddingEncodingFormat | None = None,
    user: str | None = None,
) -> EmbeddingModelResponse:
    """Asynchronously run an embedding model with a given input.

    Parameters
    ----------
    input : str | Iterable[str]
        The input to generate embeddings for.
    model : EmbeddingModelName | str
        The model to use for generating embeddings.
    provider: ModelProviderName | ModelProvider | None
        The provider to use for generating embeddings.
    base_url: str | None
        The base URL to use for generating embeddings.
    api_key: str | None
        The API key to use for generating embeddings.
    dimensions : int | None
        The dimensions to generate embeddings for.
    encoding_format : EmbeddingEncodingFormat | None
        The encoding format to use for generating embeddings.
    user : str | None
        The user to generate embeddings for.

    Returns
    -------
    EmbeddingModelResponse
        The response from the embedding model.
    """
    return await EmbeddingModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=EmbeddingModelSettings(
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        ),
    ).arun(input=input)


def run_embed(
    input: str | Iterable[str],
    model: EmbeddingModelName | str = "openai/text-embedding-3-small",
    *,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
    encoding_format: EmbeddingEncodingFormat | None = None,
    user: str | None = None,
) -> EmbeddingModelResponse:
    """Asynchronously run an embedding model with a given input.

    Parameters
    ----------
    input : str | Iterable[str]
        The input to generate embeddings for.
    model : EmbeddingModelName | str
        The model to use for generating embeddings.
    provider: ModelProviderName | ModelProvider | None
        The provider to use for generating embeddings.
    base_url: str | None
        The base URL to use for generating embeddings.
    api_key: str | None
        The API key to use for generating embeddings.
    dimensions : int | None
        The dimensions to generate embeddings for.
    encoding_format : EmbeddingEncodingFormat | None
        The encoding format to use for generating embeddings.
    user : str | None
        The user to generate embeddings for.

    Returns
    -------
    EmbeddingModelResponse
        The response from the embedding model.
    """
    return EmbeddingModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=EmbeddingModelSettings(
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        ),
    ).run(input=input)


def embedder(
    model: EmbeddingModelName | str = "openai/text-embedding-3-small",
    *,
    provider: ModelProviderName | ModelProvider | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
    encoding_format: EmbeddingEncodingFormat | None = None,
    user: str | None = None,
) -> EmbeddingModel:
    """Function factory method for creating an embedding model.

    This is genuinely one of those this doesnt need to be here methods,
    but its nice and consistent with everything else and consistency
    is key!

    Parameters
    ----------
    model : EmbeddingModelName | str
        The model to use for generating embeddings.
    provider: ModelProviderName | ModelProvider | None
        The provider to use for generating embeddings.
    base_url: str | None
        The base URL to use for generating embeddings.
    api_key: str | None
        The API key to use for generating embeddings.
    dimensions : int | None
        The dimensions to generate embeddings for.
    encoding_format : EmbeddingEncodingFormat | None
        The encoding format to use for generating embeddings.
    user : str | None
        The user to generate embeddings for.

    Returns
    -------
    EmbeddingModel
        An EmbeddingModel instance with run and arun methods pre-bound to the specified settings.
    """
    return EmbeddingModel(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        settings=EmbeddingModelSettings(
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        ),
    )
