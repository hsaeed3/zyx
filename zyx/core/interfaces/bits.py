"""zyx.core.interfaces.bits"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Literal,
    OrderedDict,
    TypeVar,
)

from httpx import URL
from openai.types.embedding import Embedding

from ..._internal._exceptions import BitError
from ..models.embeddings.model import (
    ChonkieEmbeddingModel,
    EmbeddingModel,
    EmbeddingModelName,
)
from ..models.language.model import LanguageModel, LanguageModelName
from ..models.language.types import LanguageModelResponse
from ..processing.text.text import Text, TextChunk
from .operators import Operators

if TYPE_CHECKING:
    from chonkie.chunker.base import BaseChunker
    from chonkie.embeddings.base import (
        BaseEmbeddings as ChonkieEmbeddingModel,
    )
    from instructor import Mode
    from openai.types.chat import ChatCompletionMessageParam

__all__ = (
    "Bit",
    "BitQueryResponse",
    "to_bit",
)


T = TypeVar("T")
C = TypeVar("C")
R = TypeVar("R")


@dataclass
class BitQueryResponse(Generic[R]):
    """A response when using the .query() method on a bit."""

    response: LanguageModelResponse[R]
    """The response from the language model."""

    results: List[TextChunk] | None = None
    """The chunked or relevant content returned from the query.
    
    This is only present if the content within a bit is embedded.
    """

    def __str__(self) -> str:
        from ..._internal._beautification import Beautifier

        return Beautifier.for_response(
            type_name="BitQueryResponse",
            scheme=Beautifier.Scheme(
                primary=Beautifier.Colors.DODGER_BLUE2,
                accent=Beautifier.Colors.SANDY_BROWN,
                border=Beautifier.Colors.DODGER_BLUE3,
            ),
            content=self.response.content,
            main_params={"Model": self.response.model},
            metadata={
                "Type": self.response.type.__name__
                if self.response.type
                else None,
                "Streamed": self.response.is_chunk,
                "Structured": self.response.is_structured,
            },
        ).__str__()

    def __rich__(self) -> str:
        from ..._internal._beautification import Beautifier

        return Beautifier.for_response(
            type_name="BitQueryResponse",
            scheme=Beautifier.Scheme(
                primary=Beautifier.Colors.DODGER_BLUE2,
                accent=Beautifier.Colors.SANDY_BROWN,
                border=Beautifier.Colors.DODGER_BLUE3,
            ),
            content=self.response.content,
            main_params={"Model": self.response.model},
            metadata={
                "Type": self.response.type.__name__
                if self.response.type
                else None,
                "Streamed": self.response.is_chunk,
                "Structured": self.response.is_structured,
            },
        ).rich()


class Bit(Operators[str], Generic[T]):
    """A 'bit' is a representation of any content that can be queried by a language model or represented as text.

    Bits provide a unified interface for working with different types of content (strings, files, URLs)
    and enable semantic search, chunking, and language model queries on that content.

    Parameters
    ----------
    content : T
        The source content for this bit. Can be a string, Path, URL, or other content type.
    chunk_size : int, default=1000
        The size of chunks to create when splitting the content.
    embeddings : bool, default=True
        Whether to create embeddings for the content to enable semantic search.
    embedding_model : EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
        The embedding model to use for creating embeddings.
    chunker : BaseChunker | None, default=None
        Custom chunker to use for splitting content. If None, uses default chunker.
    is_template : bool | None, default=None
        Whether the content should be treated as a template. If None, auto-infers.

    Attributes
    ----------
    content : str
        The string representation of the bit's content.
    source : T
        The original source content provided to the bit.
    text : Text
        A Text instance representation of this bit for advanced text processing.

    Examples
    --------
    >>> # Create a bit from a string
    >>> my_bit = bit("This is some content")
    >>>
    >>> # Query the bit with a language model
    >>> response = my_bit.query("What is this about?")
    >>>
    >>> # Create a bit with semantic search enabled
    >>> searchable_bit = bit("Long document content", embeddings=True)
    >>> response = searchable_bit.query("Find relevant info", search=True, top_k=5)
    """

    @cached_property
    def content(self) -> str:
        """The content of this bit."""
        return self._content

    @cached_property
    def source(self) -> T:
        """The source content of this bit."""
        return self._source

    @cached_property
    def text(self) -> Text:
        """A 'Text' instance representation of this bit."""
        return self._text

    def __init__(
        self,
        content: T,
        *,
        chunk_size: int = 1000,
        embeddings: bool = False,
        embedding_model: (
            EmbeddingModelName
            | EmbeddingModel
            | ChonkieEmbeddingModel
            | str
        ) = "openai/text-embedding-3-small",
        chunker: BaseChunker | None = None,
        is_template: bool | None = None,
        language_model: LanguageModelName
        | LanguageModel
        | str = "openai/gpt-4o-mini",
    ):
        """Initialize a 'Bit' instance from a content."""

        self._source = content

        if not isinstance(content, str):
            content = self._to_string()

        self._content = content
        self._text = Text(
            self._content,
            chunk_size=chunk_size,
            embeddings=embeddings,
            embedding_model=embedding_model,
            chunker=chunker,
            is_template=is_template,
        )
        
        # Initialize language model for LLM operations
        if isinstance(language_model, str):
            self._model = LanguageModel(language_model)
        else:
            self._model = language_model

    def _to_string(self) -> str:
        """Convert the content of this bit to a string."""
        return str(self._content)

    def _process_messages(
        self,
        messages: str | Iterable[ChatCompletionMessageParam] | None,
    ) -> List[Dict[str, Any]]:
        """Process messages into the format expected by the language model.

        Args:
            messages: String, list of message dicts, or None

        Returns:
            List of properly formatted message dicts
        """
        if messages is None:
            return []

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        # Already a list of messages
        return list(messages)

    def _compute_similarity(
        self, query_embedding: Embedding, chunk: TextChunk
    ) -> float:
        """Compute cosine similarity between query embedding and chunk embedding.

        Args:
            query_embedding: The query embedding
            chunk: The text chunk with embedding

        Returns:
            Similarity score (0-1)
        """
        import numpy as np

        # Get embedding vectors
        if hasattr(query_embedding, "embedding"):
            query_vec = np.array(query_embedding.embedding)
        else:
            query_vec = np.array(query_embedding)

        if hasattr(chunk.embedding, "embedding"):
            chunk_vec = np.array(chunk.embedding.embedding)
        else:
            chunk_vec = np.array(chunk.embedding)

        # Cosine similarity
        similarity = np.dot(query_vec, chunk_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
        )
        return float(similarity)

    def _search_chunks(
        self, query_text: str, top_k: int = 3
    ) -> List[TextChunk]:
        """Search for relevant chunks based on semantic similarity.

        Args:
            query_text: The search query
            top_k: Number of top chunks to return

        Returns:
            List of most relevant chunks
        """
        if not self._text.is_embedded or not self._text.chunks:
            return []

        # Get query embedding
        if isinstance(
            self._text._embedding_model,
            (EmbeddingModel, ChonkieEmbeddingModel),
        ):
            response = self._text._embedding_model.run(input=query_text)
            query_embedding = response.data[0]
        elif isinstance(self._text._embedding_model, str):
            model = EmbeddingModel(model=self._text._embedding_model)
            response = model.run(input=query_text)
            query_embedding = response.data[0]
        else:
            return []

        # Compute similarities
        chunk_scores = []
        for chunk in self._text.chunks:
            if chunk.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding, chunk
                )
                chunk_scores.append((chunk, similarity))

        # Sort by similarity and return top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in chunk_scores[:top_k]]

    async def aquery(
        self,
        messages: str | Iterable[ChatCompletionMessageParam] | None = None,
        *,
        type: type[R] | None = None,
        language_model: LanguageModelName
        | LanguageModel
        | str = "openai/gpt-4o-mini",
        instructions: str | None = None,
        query_strategy: Literal["direct", "llm"] = "direct",
        search: bool = False,
        top_k: int = 3,
        stream: bool = False,
        instructor_mode: Mode | None = None,
    ) -> BitQueryResponse[R]:
        """Asynchronously query the bit with a language model.

        Args:
            messages: Query message(s) - can be a string or list of messages
            type: Optional response type for structured output
            language_model: The language model to use
            instructions: Optional instructions for the model
            query_strategy: Strategy for generating search queries:
                - "direct": Use the user's message directly as search query
                - "llm": Use LLM to generate an optimized search query
            search: Whether to perform semantic search on chunks (only works if embedded)
            top_k: Number of top chunks to retrieve in search
            stream: Whether to stream the response
            instructor_mode: Instructor mode for structured output

        Returns:
            BitQueryResponse containing the model response and optional search results
        """
        # Initialize language model if needed
        if isinstance(language_model, str):
            language_model = LanguageModel(
                model=language_model,
                instructions=instructions,
            )
        elif instructions and hasattr(language_model, "instructions"):
            language_model.instructions = instructions

        # Process input messages
        processed_messages = self._process_messages(messages)

        # Determine if we should search
        should_search = (
            search and self._text.is_embedded and self._text.is_chunked
        )
        relevant_chunks = None

        if should_search:
            # Generate search query
            if query_strategy == "llm":
                # Use LLM to generate optimized search query
                query_gen_messages = processed_messages + [
                    {
                        "role": "system",
                        "content": "Based on the user's request, generate a concise search query (1-2 sentences) that would find the most relevant information in a text corpus. Return only the search query, nothing else.",
                    }
                ]

                query_response = await language_model.arun(
                    messages=query_gen_messages,
                    type=str,
                )
                search_query = query_response.content
            else:
                # Use direct message as search query
                user_messages = [
                    msg
                    for msg in processed_messages
                    if msg.get("role") == "user"
                ]
                search_query = (
                    user_messages[-1]["content"] if user_messages else ""
                )

            # Perform semantic search
            if search_query:
                relevant_chunks = self._search_chunks(
                    search_query, top_k=top_k
                )

        # Build context for the model
        context_messages = []

        # Add bit content as context
        if relevant_chunks:
            # Use only relevant chunks
            context_text = "\n\n".join(
                [
                    f"[Chunk {i + 1}]\n{chunk.text}"
                    for i, chunk in enumerate(relevant_chunks)
                ]
            )
        else:
            # Use full text
            context_text = self._text.text

        context_messages.append(
            {
                "role": "system",
                "content": f"Context information:\n\n{context_text}",
            }
        )

        # Combine with user messages
        final_messages = context_messages + processed_messages

        # Run the query
        if type:
            response = await language_model.arun(
                messages=final_messages,
                type=type,
                stream=stream,
                instructor_mode=instructor_mode,
            )
        else:
            response = await language_model.arun(
                messages=final_messages,
                stream=stream,
            )

        return BitQueryResponse(response=response, results=relevant_chunks)

    def query(
        self,
        messages: str | Iterable[ChatCompletionMessageParam] | None = None,
        *,
        type: type[R] | None = None,
        language_model: LanguageModelName
        | LanguageModel
        | str = "openai/gpt-4o-mini",
        instructions: str | None = None,
        query_strategy: Literal["direct", "llm"] = "direct",
        search: bool = False,
        top_k: int = 3,
        stream: bool = False,
        instructor_mode: Mode | None = None,
    ) -> BitQueryResponse[R]:
        """Query the bit with a language model (synchronous version).

        Args:
            messages: Query message(s) - can be a string or list of messages
            type: Optional response type for structured output
            language_model: The language model to use
            instructions: Optional instructions for the model
            query_strategy: Strategy for generating search queries:
                - "direct": Use the user's message directly as search query
                - "llm": Use LLM to generate an optimized search query
            search: Whether to perform semantic search on chunks (only works if embedded)
            top_k: Number of top chunks to retrieve in search
            stream: Whether to stream the response
            instructor_mode: Instructor mode for structured output

        Returns:
            BitQueryResponse containing the model response and optional search results
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.aquery(
                        messages=messages,
                        type=type,
                        language_model=language_model,
                        instructions=instructions,
                        query_strategy=query_strategy,
                        search=search,
                        top_k=top_k,
                        stream=stream,
                        instructor_mode=instructor_mode,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.aquery(
                    messages=messages,
                    type=type,
                    language_model=language_model,
                    instructions=instructions,
                    query_strategy=query_strategy,
                    search=search,
                    top_k=top_k,
                    stream=stream,
                    instructor_mode=instructor_mode,
                )
            )

    def _get_content_for_llm(self) -> str | None:
        """Get the content formatted for LLM operations.
        
        Returns
        -------
        str | None
            The content as a string, or None if no content.
        """
        return self._content if self._content else None

    def _get_model_for_llm(self) -> LanguageModel:
        """Get the language model for LLM operations.
        
        Returns
        -------
        LanguageModel
            The language model to use.
        """
        return self._model

    def __str__(self) -> str:
        return self.content

    def __rich__(self) -> str:
        from rich.console import Group
        from rich.markup import escape
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text as RichText

        if self.text.kind == "string":
            title = (
                f"[bold]Bit[bold] | {type(self.source).__name__.upper()}"
            )
        elif (
            self.text.kind == "file:simple"
            or self.text.kind == "file:complex"
        ):
            title = f"[bold]Bit[bold] | Path: {escape(str(self.source))}"
        elif self.text.kind == "url":
            title = f"[bold]Bit[bold] | URL: {escape(str(self.source))}"

        if self.text.is_template:
            title += " [green](Is Template)[/green]"
        else:
            title += " [red](Not Template)[/red]"

        syntax = Syntax(self.content, "markdown", word_wrap=True)

        return Panel(
            syntax,
            title=title,
            border_style="blue",
        )


def to_bit(
    content: T | Path | URL,
    *,
    chunk_size: int = 1000,
    embeddings: bool = False,
    embedding_model: (
        EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
    ) = "openai/text-embedding-3-small",
    chunker: BaseChunker | None = None,
    is_template: bool | None = None,
    language_model: LanguageModelName
    | LanguageModel
    | str = "openai/gpt-4o-mini",
) -> Bit[T]:
    """Create a new 'bit' from a source of content. A bit a representation of any content that can be
    queried by a language model or represented as text.

    Parameters
    ----------
    content: T | Path | URL
        The content to create the bit from. Can be a string, path to a file, or a URL.
    chunk_size: int
        The size of the chunks to create.
    embeddings: bool
        Whether to create embeddings for the content.
    embedding_model: EmbeddingModelName | EmbeddingModel | ChonkieEmbeddingModel | str
        The embedding model to use.
    chunker: BaseChunker | None
        The chunker to use.
    is_template: bool | None
        Whether the content is a template. If None, it will auto infer.
    language_model: LanguageModelName | LanguageModel | str
        The language model to use for LLM-powered operations (comparisons, queries, etc.).

    Returns
    -------
    Bit[T]
        A new 'bit' instance.
    """
    return Bit(
        content=content,
        chunk_size=chunk_size,
        embeddings=embeddings,
        embedding_model=embedding_model,
        chunker=chunker,
        is_template=is_template,
        language_model=language_model,
    )
