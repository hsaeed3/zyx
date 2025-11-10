"""zyx.core.models.embeddings.model"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingModelResponse,
)
from openai.types.embedding import Embedding

from ..definition import ModelDefinition
from ..providers import ModelProvider, ModelProviderName
from .types import (
    EmbeddingEncodingFormat,
    EmbeddingModelName,
    EmbeddingModelSettings,
)

if TYPE_CHECKING:
    from chonkie.embeddings.base import (
        BaseEmbeddings as ChonkieEmbeddingModel,
    )

__all__ = [
    "EmbeddingModel",
    "ChonkieEmbeddingModel",
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
            if model.startswith("openai/") or model.startswith(
                "text-embedding-"
            ):
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
            dimensions
            if dimensions is not None
            else self.settings.dimensions
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


class ChonkieEmbeddingModel:
    """Adapter class that wraps a Chonkie BaseEmbeddings model to make it compatible
    with the zyx EmbeddingModel interface.

    This adapter allows seamless use of Chonkie embedding models throughout the zyx
    library while maintaining compatibility with OpenAI-style embedding responses.
    """

    @property
    def kind(self) -> str:
        return "chonkie_embedding_model"

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._chonkie_model.dimension

    @property
    def chonkie_model(self) -> ChonkieEmbeddingModel:
        """Get the underlying Chonkie embedding model."""
        return self._chonkie_model

    def __init__(
        self,
        chonkie_model: ChonkieEmbeddingModel,
        model_name: str | None = None,
    ) -> None:
        """Initialize the adapter with a Chonkie embedding model.

        Parameters
        ----------
        chonkie_model : ChonkieEmbeddingModel
            The Chonkie BaseEmbeddings instance to wrap.
        model_name : str | None
            Optional model name for identification. If None, uses the model's class name.
        """
        self._chonkie_model = chonkie_model
        self._model_name = model_name or chonkie_model.__class__.__name__

    def _embeddings_to_response(
        self, embeddings: list, input_texts: list[str]
    ) -> EmbeddingModelResponse:
        """Convert numpy embeddings to OpenAI-style embedding response.

        Parameters
        ----------
        embeddings : list
            List of numpy arrays containing embeddings.
        input_texts : list[str]
            The input texts that were embedded.

        Returns
        -------
        EmbeddingModelResponse
            OpenAI-style embedding response object.
        """
        import numpy as np

        # Convert embeddings to the format expected by OpenAI's Embedding type
        embedding_objects = []
        for idx, emb in enumerate(embeddings):
            # Convert numpy array to list if needed
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb)

            # Create Embedding object
            embedding_obj = Embedding(
                embedding=emb_list, index=idx, object="embedding"
            )
            embedding_objects.append(embedding_obj)

        # Calculate total tokens (rough estimate based on text length)
        total_tokens = sum(len(text.split()) for text in input_texts)

        # Create the response object
        from openai.types.create_embedding_response import Usage

        response = EmbeddingModelResponse(
            data=embedding_objects,
            model=self._model_name,
            object="list",
            usage=Usage(
                prompt_tokens=total_tokens, total_tokens=total_tokens
            ),
        )

        return response

    async def arun(
        self, input: str | Iterable[str], **kwargs
    ) -> EmbeddingModelResponse:
        """Asynchronously generate embeddings for the input.

        Note: This wraps the synchronous Chonkie embedding call in an async context.

        Parameters
        ----------
        input : str | Iterable[str]
            The text(s) to embed.
        **kwargs
            Additional keyword arguments (ignored for Chonkie models).

        Returns
        -------
        EmbeddingModelResponse
            OpenAI-style embedding response.
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.run, input)
        return result

    def run(
        self, input: str | Iterable[str], **kwargs
    ) -> EmbeddingModelResponse:
        """Generate embeddings for the input.

        Parameters
        ----------
        input : str | Iterable[str]
            The text(s) to embed.
        **kwargs
            Additional keyword arguments (ignored for Chonkie models).

        Returns
        -------
        EmbeddingModelResponse
            OpenAI-style embedding response.
        """
        # Handle single string input
        if isinstance(input, str):
            embeddings = [self._chonkie_model.embed(input)]
            input_texts = [input]
        else:
            # Handle iterable of strings
            input_texts = list(input)
            embeddings = self._chonkie_model.embed_batch(input_texts)

        return self._embeddings_to_response(embeddings, input_texts)

    def __repr__(self) -> str:
        return f"ChonkieEmbeddingModelAdapter(model={self._model_name}, dimensions={self.dimensions})"
