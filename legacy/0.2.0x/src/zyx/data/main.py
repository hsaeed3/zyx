# zyx =========================================================================

from typing import List, Optional, Union, Literal, Callable
from sqlmodel import Field, SQLModel
import os

DBType = Literal["sql", "qdrant"]

# --- Database Models --------------------------------------------------------------


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    filename: str
    hash: str


# --- Database Class --------------------------------------------------------------


class DB:
    """Base class for database operations."""

    def __init__(
        self,
        db_type: DBType = "sql",
        db_path: str = ":memory:",
        embedding_model: str = "text-embedding-ada-002",
        vector_size: int = 1536,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
        chunk_size: int = 1000,
        token_counter: Optional[Callable[[str], int]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.db_type = db_type
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.vector_size = vector_size
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.token_counter = token_counter
        self.api_key = api_key
        self.api_base = api_base

        if db_type == "sql":
            self._setup_sql_db()
        elif db_type == "qdrant":
            self._setup_qdrant_db(qdrant_url)

    def _setup_sql_db(self):
        from sqlalchemy.engine import create_engine

        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)

    def _setup_qdrant_db(self, qdrant_url: str):
        from qdrant_client.models import VectorParams, Distance
        from qdrant_client.qdrant_client import QdrantClient

        self.qdrant_client = QdrantClient(qdrant_url)
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )

    def add(self, path: Union[str, List[str]]):
        """Add a document or a list of documents to the database.
        Utilizes chunking to split the document into smaller chunks.

        Args:
            path (Union[str, List[str]]): The path to the document file or a list of paths.

        Raises:
            ValueError: If the path is invalid.
        """
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        for file_path in paths:
            contents = self._read_files(file_path)
            chunked_contents = self._chunk_contents(contents)
            if self.db_type == "sql":
                self._add_to_sql(chunked_contents)
            elif self.db_type == "qdrant":
                self._add_to_qdrant(chunked_contents)

    def _read_files(self, path: str) -> List[tuple]:
        if os.path.isfile(path):
            return [self._process_file(path)]
        elif os.path.isdir(path):
            contents = []
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    contents.append(self._process_file(file_path))
            return contents
        else:
            raise ValueError(f"Invalid path: {path}")

    def _process_file(self, file_path: str) -> tuple:
        import hashlib

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        filename = os.path.basename(file_path)
        file_hash = hashlib.md5(content.encode()).hexdigest()
        return (content, filename, file_hash)

    def _chunk_contents(self, contents: List[tuple]) -> List[tuple]:
        import hashlib
        from ..core.text import chunk

        chunked_contents = []
        for content, filename, file_hash in contents:
            chunks = chunk(content, self.chunk_size, self.token_counter)
            for i, chunk_content in enumerate(chunks[0]):
                chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                chunked_contents.append(
                    (chunk_content, f"{filename}_chunk_{i}", chunk_hash)
                )
        return chunked_contents

    def _add_to_sql(self, contents: List[tuple]):
        from sqlalchemy.orm import Session
        from sqlalchemy.sql.expression import select

        with Session(self.engine) as session:
            for content, filename, file_hash in contents:
                existing_doc = session.execute(
                    select(Document).where(Document.hash == file_hash)
                ).first()
                if not existing_doc:
                    document = Document(
                        content=content, filename=filename, hash=file_hash
                    )
                    session.add(document)
            session.commit()

    def _add_to_qdrant(self, contents: List[tuple]):
        from qdrant_client.models import PointStruct
        from ..ai import embeddings

        points = []
        contents_list = [content for content, _, _ in contents]
        embedding_response = embeddings(
            inputs=contents_list,
            model=self.embedding_model,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        for i, ((content, filename, file_hash), embedding_data) in enumerate(
            zip(contents, embedding_response["data"])
        ):
            embedding_vector = embedding_data["embedding"]
            point = PointStruct(
                id=i,
                vector=embedding_vector,
                payload={"content": content, "filename": filename, "hash": file_hash},
            )
            points.append(point)

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Searches the database for documents similar to the query.

        Args:
            query (str): The query string.
            top_k (int, optional): The number of results to return. Defaults to 5.

        Returns:
            List[dict]: A list of dictionaries containing the content and filename.
        """
        if self.db_type == "sql":
            return self._search_sql(query, top_k)
        elif self.db_type == "qdrant":
            return self._search_qdrant(query, top_k)

    def _search_sql(self, query: str, top_k: int) -> List[dict]:
        from sqlalchemy.orm import Session
        from sqlalchemy.sql.expression import select

        with Session(self.engine) as session:
            statement = (
                select(Document).where(Document.content.contains(query)).limit(top_k)
            )
            results = session.execute(statement).all()
            return [
                {"content": doc.content, "filename": doc.filename} for doc in results
            ]

    def _search_qdrant(self, query: str, top_k: int) -> List[dict]:
        from ..ai import embeddings

        embedding_response = embeddings(
            inputs=[query],
            model=self.embedding_model,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        query_vector = embedding_response["data"][0]["embedding"]
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        return [
            {"content": hit.payload["content"], "filename": hit.payload["filename"]}
            for hit in search_result
        ]


# ==============================================================================


def db(
    db_type: DBType = "sql",
    db_path: str = ":memory:",
    embedding_model: str = "text-embedding-ada-002",
    vector_size: int = 1536,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "documents",
    chunk_size: int = 1000,
    token_counter: Optional[Callable[[str], int]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> DB:
    """
    Creates and returns a DB instance with the specified configuration.

    Example:
        ```python
        db = DB()

        # Add a document
        db.add("path/to/document.txt")

        # Search for similar documents
        results = db.search("query")
        ```

    Parameters:
        db_type (DBType): The type of database to use ("sql" or "qdrant").
        db_path (str): The path to the database file (for SQL) or ":memory:" for in-memory database.
        embedding_model (str): The name of the embedding model to use.
        vector_size (int): The size of the vector embeddings.
        qdrant_url (str): The URL of the Qdrant server (for Qdrant database).
        collection_name (str): The name of the collection in Qdrant.
        chunk_size (int): The maximum number of tokens per chunk.
        token_counter (Callable[[str], int], optional): A function to count tokens in a string.
        api_key (str, optional): API key for the embedding model.
        api_base (str, optional): API base URL for the embedding model.

    Returns:
        DB: An instance of the DB class.
    """
    return DB(
        db_type=db_type,
        db_path=db_path,
        embedding_model=embedding_model,
        vector_size=vector_size,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        chunk_size=chunk_size,
        token_counter=token_counter,
        api_key=api_key,
        api_base=api_base,
    )


# ==============================================================================
