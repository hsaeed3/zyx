try:
    from chromadb import Client
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from pydantic import BaseModel, Field
    from typing import Callable, List, Union, Optional, Literal, Type
    import uuid
    import numpy as np

    from ...lib.types.document import Document
    from ...lib.utils.logger import get_logger
    from ...client import completion, InstructorMode
    from ..data.chunk import chunk
    from ..completions.base.generate import generate
except ImportError:
    import os

    print(
        "The [bold]`zyx(data)`[/bold] data extension is required to use this module. Install it?"
    )
    if input("Install? (y/n)") == "y":
        os.system("pip install 'zyx[data]'")
        exit(1)
    else:
        print("Exiting...")
        exit(1)


logger = get_logger("memory")


class ChromaNode(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[ChromaNode] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    summary: str


class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Custom embedding logic using OpenAI API
        embeddings = []
        for text in texts:
            embedding = self.get_embedding_from_api(text)  # Replace with actual call
            embeddings.append(embedding)
        return embeddings

    def get_embedding_from_api(self, text: str) -> List[float]:
        # Simulate embedding API call (you should implement the actual API call here)
        return np.random.randn(1536).tolist()  # Dummy embedding for example


class Memory:
    """
    Class for storing and retrieving data using Chroma.
    """

    def __init__(
        self,
        collection_name: str = "my_collection",
        model_class: Optional[Type[BaseModel]] = None,
        embedding_api_key: Optional[str] = None,
        location: Union[Literal[":memory:"], str] = ":memory:",
        persist_directory: str = "chroma_db",
        chunk_size: int = 512,
        model: str = "gpt-4o-mini",
    ):
        """
        Class for storing and retrieving data using Chroma.

        Args:
            collection_name (str): The name of the collection.
            model_class (Type[BaseModel], optional): Model class for storing data.
            embedding_api_key (str, optional): API key for embedding model.
            location (str): ":memory:" for in-memory database or a string path for persistent storage.
            persist_directory (str): Directory for persisting Chroma database (if not using in-memory storage).
            chunk_size (int): Size of chunks for text splitting.
            model (str): Model name for text summarization.
        """

        self.collection_name = collection_name
        self.embedding_api_key = embedding_api_key
        self.model_class = model_class
        self.location = location
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.model = model

        self.client = self._initialize_client()
        self.collection = self._create_or_get_collection()

    def _initialize_client(self):
        """
        Initialize Chroma client. Use in-memory database if location is ":memory:",
        otherwise, use persistent storage at the specified directory.
        """
        if self.location == ":memory:":
            logger.info("Using in-memory Chroma storage.")
            return Client()  # In-memory by default
        else:
            logger.info(f"Using persistent Chroma storage at {self.persist_directory}.")
            settings = Settings(persist_directory=self.persist_directory)
            return Client(settings)

    def _create_or_get_collection(self):
        """Retrieve or create a Chroma collection with a custom embedding function."""
        embedding_fn = CustomEmbeddingFunction(api_key=self.embedding_api_key)
        if self.collection_name in self.client.list_collections():
            logger.info(f"Collection '{self.collection_name}' already exists.")
            return self.client.get_collection(
                self.collection_name, embedding_function=embedding_fn
            )
        else:
            logger.info(f"Creating collection '{self.collection_name}'.")
            return self.client.create_collection(
                name=self.collection_name, embedding_function=embedding_fn
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a given text using the custom embedding function.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The embedding for the text.
        """
        embedding_fn = CustomEmbeddingFunction(api_key=self.embedding_api_key)
        return embedding_fn([text])[0]  # Return the first (and only) embedding

    def add(
        self,
        data: Union[str, List[str], Document, List[Document]],
        metadata: Optional[dict] = None,
    ):
        """Add documents or data to Chroma.

        Args:
            data (Union[str, List[str], Document, List[Document]]): The data to add to Chroma.
            metadata (Optional[dict]): The metadata to add to the data.
        """
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, Document):
            data = [data]

        ids, embeddings, texts, metadatas = [], [], [], []

        for item in data:
            try:
                if isinstance(item, Document):
                    text = item.content
                    metadata = item.metadata
                else:
                    text = item

                # Chunk the content
                chunks = chunk(text, chunk_size=self.chunk_size, model=self.model)

                for chunk_text in chunks:
                    embedding_vector = self._get_embedding(chunk_text)
                    ids.append(str(uuid.uuid4()))
                    embeddings.append(embedding_vector)
                    texts.append(chunk_text)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["chunk"] = True
                    metadatas.append(chunk_metadata)
            except Exception as e:
                logger.error(f"Error processing item: {item}. Error: {e}")

        if embeddings:
            try:
                # Ensure metadatas is not empty
                metadatas = [m if m else {"default": "empty"} for m in metadatas]
                self.collection.add(
                    ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
                )
                logger.info(
                    f"Successfully added {len(embeddings)} chunks to the collection."
                )
            except Exception as e:
                logger.error(f"Error adding points to collection: {e}")
        else:
            logger.warning("No valid embeddings to add to the collection.")

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        """Search in Chroma collection.

        Args:
            query (str): The query to search for.
            top_k (int): The number of results to return.

        Returns:
            SearchResponse: The search results.
        """
        try:
            query_embedding = self._get_embedding(query)
            search_results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )

            nodes = []
            for i in range(len(search_results["ids"][0])):  # Note the [0] here
                node = ChromaNode(
                    id=search_results["ids"][0][i],
                    text=search_results["documents"][0][i],
                    embedding=query_embedding,
                    metadata=search_results["metadatas"][0][i]
                    if search_results["metadatas"]
                    else {},
                )
                nodes.append(node)
            return SearchResponse(query=query, results=nodes)
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return SearchResponse(query=query)  # Return empty results on error

    def _summarize_results(self, results: List[ChromaNode]) -> str:
        """Summarize the search results.

        Args:
            results (List[ChromaNode]): The search results.

        Returns:
            str: The summary of the search results.
        """

        class SummaryModel(BaseModel):
            summary: str

        texts = [node.text for node in results]
        combined_text = "\n\n".join(texts)

        summary = generate(
            SummaryModel,
            instructions="Provide a concise summary of the following text, focusing on the most important information:",
            model=self.model,
            n=1,
        )

        return summary.summary

    def completion(
        self,
        messages: Union[str, List[dict]] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = 5,
        tools: Optional[List[Union[Callable, dict, BaseModel]]] = None,
        run_tools: Optional[bool] = True,
        response_model: Optional[BaseModel] = None,
        mode: Optional[InstructorMode] = "tool_call",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = 3,
        verbose: Optional[bool] = False,
    ):
        """Perform completion with context from Chroma.

        Args:
            messages (Union[str, List[dict]]): The messages to use for the completion.
            model (Optional[str]): The model to use for the completion.
            top_k (Optional[int]): The number of results to return from the search.
            tools (Optional[List[Union[Callable, dict, BaseModel]]]): The tools to use for the completion.
            run_tools (Optional[bool]): Whether to run the tools for the completion.
            response_model (Optional[BaseModel]): The response model to use for the completion.
            mode (Optional[InstructorMode]): The mode to use for the completion.
            base_url (Optional[str]): The base URL to use for the completion.
            api_key (Optional[str]): The API key to use for the completion.
            organization (Optional[str]): The organization to use for the completion.
            top_p (Optional[float]): The top p to use for the completion.
            temperature (Optional[float]): The temperature to use for the completion.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            max_retries (Optional[int]): The maximum number of retries to use for the completion.
            verbose (Optional[bool]): Whether to print the messages to the console.
        """
        logger.info(f"Initial messages: {messages}")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            messages = [
                {"role": "user", "content": m} if isinstance(m, str) else m
                for m in messages
            ]

        query = messages[-1].get("content", "") if messages else ""

        try:
            results = self.search(query, top_k=top_k)
            summarized_results = self._summarize_results(results.results)
        except Exception as e:
            logger.error(f"Error during search or summarization: {e}")
            summarized_results = ""

        if messages:
            if not any(message.get("role", "") == "system" for message in messages):
                system_message = {
                    "role": "system",
                    "content": f"Relevant information retrieved: \n {summarized_results}",
                }
                messages.insert(0, system_message)
            else:
                for message in messages:
                    if message.get("role", "") == "system":
                        message["content"] += (
                            f"\nAdditional context: {summarized_results}"
                        )

        try:
            result = completion(
                messages=messages,
                model=model or self.model,
                tools=tools,
                run_tools=run_tools,
                response_model=response_model,
                mode=mode,
                base_url=base_url,
                api_key=api_key,
                organization=organization,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

            if verbose:
                logger.info(f"Completion result: {result}")

            return result
        except Exception as e:
            logger.error(f"Error during completion: {e}")
            raise


if __name__ == "__main__":
    try:
        # Initialize the Store
        store = Memory(
            collection_name="test_collection", embedding_api_key="your-api-key"
        )

        # Test adding single string
        store.add("This is a single string test.")
        print("Added single string.")

        # Test adding list of strings
        store.add(
            [
                "Multiple string test 1",
                "Multiple string test 2",
                "Multiple string test 3",
            ]
        )
        print("Added multiple strings.")

        # Test adding Document
        doc = Document(content="Document test content", metadata={"source": "test"})
        store.add(doc)
        print("Added Document.")

        # Test adding list of Documents
        docs = [
            Document(content="Document 1 content", metadata={"source": "test1"}),
            Document(content="Document 2 content", metadata={"source": "test2"}),
        ]
        store.add(docs)
        print("Added multiple Documents.")

        # Test search
        search_query = "test"
        results = store.search(search_query, top_k=3)
        print(f"\nSearch results for '{search_query}':")
        for result in results.results:
            print(f"ID: {result.id}, Text: {result.text}, Metadata: {result.metadata}")

        # Test search with more results than in collection
        large_k_results = store.search(search_query, top_k=100)
        print(f"\nSearch results with large top_k (100):")
        print(f"Number of results returned: {len(large_k_results.results)}")

        # Test completion
        completion_query = "What is the main topic of the documents?"
        completion_result = store.completion(completion_query)
        print(f"\nCompletion result for '{completion_query}':")
        print(completion_result)

        # Test completion with custom model and parameters
        custom_completion = store.completion(
            messages=[{"role": "user", "content": "Summarize the documents."}],
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
        )
        print("\nCustom completion result:")
        print(custom_completion)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
