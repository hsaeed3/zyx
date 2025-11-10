"""zyx.core.processing.text.text"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, List, OrderedDict, TypeVar

from httpx import URL
from openai.types.embedding import Embedding

from ...._internal._exceptions import ChunkingError, TextError
from ...models.embeddings.model import (
    ChonkieEmbeddingModel,
    EmbeddingModel,
    EmbeddingModelName,
)
from ...processing.text.chunking import _get_chunker, to_chunks
from ...processing.text.ingestion import (
    StringKind,
    identify_string_kind,
    ingest_file_as_string,
    ingest_url_as_string,
    string_has_template_syntax,
)
from ...processing.text.templating import render_template_string

if TYPE_CHECKING:
    from chonkie.chunker.base import BaseChunker
    from chonkie.types import Chunk


@dataclass
class TextChunk:
    """Internal class used to represent both a chunk of text and it's
    embedding"""

    text: str
    """The text content of this chunk."""

    embedding: Embedding | None
    """The embedding of this chunk."""

    index: int
    """The index of this chunk."""

    start: int
    """The start index of this chunk."""

    end: int
    """The end index of this chunk."""

    length: int
    """The length of this chunk."""


class Text:
    """An incredibly modularized class that handles various forms of text ingestion,
    processing & access.
    """

    @cached_property
    def text(self) -> str:
        """The raw text content of this text."""
        return self._text

    @property
    def kind(self) -> StringKind:
        """The kind of text content of this text."""
        return self._kind

    @property
    def chunks(self) -> List[TextChunk]:
        """The chunks of text content of this text."""
        return self._chunks

    @property
    def is_chunked(self) -> bool:
        """Whether the text has been chunked."""
        return self._is_chunked

    @property
    def is_embedded(self) -> bool:
        """Whether the text has been embedded."""
        return self._is_embedded

    @property
    def is_template(self) -> bool:
        """Whether the text is a template."""
        return self._is_template

    @property
    def embeddings(self) -> List[Embedding]:
        """The embeddings of the text."""
        return self._embeddings

    def __init__(
        self,
        text: str,
        *,
        chunk_size: int = 1000,
        embeddings: bool = True,
        embedding_model: (
            EmbeddingModelName
            | EmbeddingModel
            | ChonkieEmbeddingModel
            | str
        ) = "openai/text-embedding-3-small",
        chunker: BaseChunker | None = None,
        # added just to make life easier for this class
        # if none it'll auto infer (only simple files and strings)
        is_template: bool | None = None,
    ) -> None:
        """Initialize a Text instance from a string, filepath or a URL.

        Parameters
        ----------
        text : str
            The content to initialize the Text instance from.
        chunk_size : int
            The size of the chunks to create.
        embeddings : bool
            Whether to create embeddings for the text.
        embedding_model : EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
            The embedding model to use.
        chunker : BaseChunker | None
            The chunker to use.
        is_template : bool | None
            Whether the content is a template. If None, it will auto infer.
        """
        self._chunk_size = chunk_size
        self._do_embeddings = embeddings
        self._chunker = chunker
        self._is_template = is_template

        self._is_chunked: bool = False
        self._is_embedded: bool = False
        self._kind: StringKind | None = None
        self._chunks: List[TextChunk] = []
        self._embeddings: List[Embedding] = []

        # Detect if we're using a ChonkieEmbeddingModel and wrap it in adapter
        self._using_chonkie_embeddings = False
        self._chonkie_model_raw = (
            None  # Store raw chonkie model for chunker
        )

        try:
            if isinstance(embedding_model, ChonkieEmbeddingModel):
                self._using_chonkie_embeddings = True
                self._chonkie_model_raw = embedding_model
                # Wrap in adapter for unified interface
                self._embedding_model = ChonkieEmbeddingModel(
                    chonkie_model=embedding_model
                )
            else:
                self._embedding_model = embedding_model
        except ImportError:
            self._embedding_model = embedding_model

        # assume its not a link or a file
        if len(text) > 150:
            self._kind = "string"
        if not self._kind:
            self._kind = identify_string_kind(text)

        if self._kind == "file:simple":
            if self._is_template is None:
                self._is_template = string_has_template_syntax(text)
            self._text = ingest_file_as_string(text)

        elif self._kind == "file:complex":
            self._text = ingest_file_as_string(text)

        elif self._kind == "url":
            self._text = ingest_url_as_string(text)
        else:
            self._text = text

        # Auto-chunk and embed if necessary
        if len(self._text) > self._chunk_size:
            self.chunk()
            if self._do_embeddings:
                self.embed()

    def render(
        self,
        context: Any | None = None,
        jinja_config: dict | None = None,
    ) -> str:
        """Render the text with a given context and config.

        Parameters
        ----------
        context : Any | None
            The context to render the text with.
        config : dict | None
            The Jinja2 config to render the text with.

        Returns
        -------
        str
            The rendered text.
        """
        if not self._is_template:
            raise TextError("Text is not a template")

        try:
            return render_template_string(
                self._text,
                context=context,
                config=jinja_config,
            )
        except Exception as e:
            raise TextError(f"Failed to render text: {e}") from e

    def chunk(self):
        """Chunk the text into smaller chunks. This operation can only be
        performed once per instance.
        """
        if self._is_chunked:
            raise ChunkingError(f"Text has already been chunked")

        if not len(self._text) > self._chunk_size:
            self._is_chunked = True
            return

        try:
            # Use SemanticChunker if we have a ChonkieEmbeddingModel
            if self._using_chonkie_embeddings and self._chonkie_model_raw:
                from chonkie.chunker.semantic import SemanticChunker

                semantic_chunker = SemanticChunker(
                    embedding_model=self._chonkie_model_raw,  # Use raw model for chunker
                    chunk_size=self._chunk_size,
                )
                chunks: List[Chunk] = semantic_chunker.chunk(self._text)
            else:
                # Use default chunker
                chunks: List[Chunk] = to_chunks(
                    self._text, self._chunk_size, self._chunker
                )

            self._chunks: List[TextChunk] = [
                TextChunk(
                    text=chunk.text,
                    embedding=chunk.embedding,
                    index=index,
                    start=chunk.start_index,
                    end=chunk.end_index,
                    length=len(chunk.text),
                )
                for index, chunk in enumerate(chunks)
            ]
        except Exception as e:
            raise ChunkingError(f"Failed to chunk text: {e}") from e

        if not self._chunks:
            raise ChunkingError(
                f"Failed to chunk text, or no chunks were created"
            )

        self._is_chunked = True

    def embed(self):
        """Generate embeddings for the text or its chunks.

        If using a ChonkieEmbeddingModel with SemanticChunker, embeddings
        are already generated during chunking. Otherwise, we generate them
        using the provided embedding model.
        """
        if self._is_embedded:
            raise TextError("Text has already been embedded")

        try:
            # If using chonkie embeddings and chunks exist, check if embeddings are already set
            if self._using_chonkie_embeddings and self._is_chunked:
                # Check if embeddings are already in the chunks from SemanticChunker
                has_embeddings = any(
                    chunk.embedding is not None for chunk in self._chunks
                )

                if has_embeddings:
                    # Convert numpy embeddings to OpenAI Embedding objects
                    import numpy as np

                    embedding_objects = []

                    for idx, chunk in enumerate(self._chunks):
                        if chunk.embedding is not None:
                            # Convert numpy array to list if needed
                            if isinstance(chunk.embedding, np.ndarray):
                                emb_list = chunk.embedding.tolist()
                            elif isinstance(chunk.embedding, list):
                                emb_list = chunk.embedding
                            else:
                                emb_list = list(chunk.embedding)

                            # Create Embedding object
                            embedding_obj = Embedding(
                                embedding=emb_list,
                                index=idx,
                                object="embedding",
                            )
                            embedding_objects.append(embedding_obj)
                            # Update chunk with proper Embedding object
                            self._chunks[idx].embedding = embedding_obj

                    self._embeddings = embedding_objects
                    self._is_embedded = True
                    return

                # If no embeddings in chunks, fall through to generate them
                else:
                    # Use the adapter to generate embeddings
                    chunk_texts = [chunk.text for chunk in self._chunks]
                    response = self._embedding_model.run(input=chunk_texts)

                    # Update chunks with embeddings and store them
                    for i, embedding_obj in enumerate(response.data):
                        self._chunks[i].embedding = embedding_obj

                    self._embeddings = [emb for emb in response.data]
                    self._is_embedded = True
                    return

            # If we have chunks, embed each chunk
            if self._is_chunked and self._chunks:
                if isinstance(
                    self._embedding_model,
                    (EmbeddingModel, ChonkieEmbeddingModel),
                ):
                    # Use our EmbeddingModel or adapter
                    chunk_texts = [chunk.text for chunk in self._chunks]
                    response = self._embedding_model.run(input=chunk_texts)

                    # Update chunks with embeddings and store them
                    for i, embedding_obj in enumerate(response.data):
                        self._chunks[i].embedding = embedding_obj

                    self._embeddings = [emb for emb in response.data]

                elif isinstance(self._embedding_model, str):
                    # Create an EmbeddingModel from the string
                    model = EmbeddingModel(model=self._embedding_model)
                    chunk_texts = [chunk.text for chunk in self._chunks]
                    response = model.run(input=chunk_texts)

                    # Update chunks with embeddings and store them
                    for i, embedding_obj in enumerate(response.data):
                        self._chunks[i].embedding = embedding_obj

                    self._embeddings = [emb for emb in response.data]

                else:
                    raise TextError(
                        f"Unsupported embedding model type: {type(self._embedding_model)}"
                    )

            else:
                # No chunks, embed the full text
                if isinstance(
                    self._embedding_model,
                    (EmbeddingModel, ChonkieEmbeddingModel),
                ):
                    response = self._embedding_model.run(input=self._text)
                    self._embeddings = [emb for emb in response.data]

                elif isinstance(self._embedding_model, str):
                    model = EmbeddingModel(model=self._embedding_model)
                    response = model.run(input=self._text)
                    self._embeddings = [emb for emb in response.data]

                else:
                    raise TextError(
                        f"Unsupported embedding model type: {type(self._embedding_model)}"
                    )

            self._is_embedded = True

        except Exception as e:
            raise TextError(f"Failed to generate embeddings: {e}") from e

    def __str__(self) -> str:
        return self.text

    def __rich__(self) -> Any:
        from rich.panel import Panel
        from rich.syntax import Syntax

        lexer = "html" if self.kind == "url" else "markdown"
        syntax = Syntax(self.text, lexer=lexer, word_wrap=True)
        title = f"Text ({self.kind}) | Chunks: {len(self.chunks) if self.is_chunked else 'N/A'} | Embedded: {self.is_embedded}"
        return Panel(syntax, title=title, expand=True, border_style="cyan")


def to_text(
    content: str | Path | URL,
    *,
    chunk_size: int = 1000,
    embeddings: bool = True,
    embedding_model: (
        EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
    ) = "openai/text-embedding-3-small",
    chunker: BaseChunker | None = None,
    is_template: bool | None = None,
) -> Text:
    """Create a Text instance from a string, filepath or a URL.

    Parameters
    ----------
    content : str | Path | URL
        The content to create the Text instance from.
    chunk_size : int
        The size of the chunks to create.
    embeddings : bool
        Whether to create embeddings for the text.
    embedding_model : EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
        The embedding model to use.
    chunker : BaseChunker | None
        The chunker to use.
    is_template : bool | None
        Whether the content is a template. If None, it will auto infer.

    Returns
    -------
    Text
        A new Text instance.
    """
    return Text(
        content=content,
        chunk_size=chunk_size,
        embeddings=embeddings,
        embedding_model=embedding_model,
        chunker=chunker,
        is_template=is_template,
    )
