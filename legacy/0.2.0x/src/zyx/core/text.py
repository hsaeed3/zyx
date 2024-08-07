# zyx ==============================================================================

__all__ = ["reader", "chunk"]

from .decorators import batch
import os
from typing import Callable, List, Union


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_docx_file(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def read_pdf_file(file_path: str) -> str:
    import fitz

    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return read_text_file(file_path)
    elif ext == ".docx":
        return read_docx_file(file_path)
    elif ext == ".pdf":
        return read_pdf_file(file_path)
    else:
        try:
            return read_text_file(file_path)
        except:
            raise ValueError(f"Unsupported file format: {ext}")


@batch(batch_size=10, verbose=True, timeout=None)
async def read_files(file_paths: List[str]) -> List[str]:
    return [read_file(file_path) for file_path in file_paths]


def reader(path: str) -> Union[str, List[str]]:
    import asyncio

    if os.path.isfile(path):
        return read_file(path)
    elif os.path.isdir(path):
        file_paths = [os.path.join(path, file_name) for file_name in os.listdir(path)]
        return asyncio.run(read_files(file_paths))
    else:
        raise ValueError(f"Invalid path: {path}")


# ==============================================================================

# zyx ==============================================================================


async def chunk_text(
    text: Union[str, List[str]],
    chunk_size: int,
    token_counter: Callable[[str], int] = None,
    memoize: bool = True,
    verbose: bool = False,
) -> Union[List[str], List[List[str]]]:
    """Split a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

    Args:
        text (Union[str, List[str]]): The text to be chunked.
        chunk_size (int): The maximum number of tokens a chunk may contain.
        token_counter (Callable[[str], int]): A callable that takes a string and returns the number of tokens in it.
        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to `False`.

    Returns:
        Union[List[str], List[List[str]]]: A list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed.
    """
    import semchunk
    import tiktoken

    if token_counter is None:
        encoding = tiktoken.encoding_for_model("gpt-4")
        token_counter = lambda x: len(encoding.encode(x))

    try:
        chunker = semchunk.chunkerify(token_counter, chunk_size, memoize=memoize)
    except ImportError:
        raise ImportError(
            "The semchunk package is required for this function. Please install it via `pip install semchunk`."
        )

    async def process_texts(
        texts: List[str],
    ) -> List[Union[List[str], List[List[str]]]]:
        return [chunker(t) for t in texts]

    if isinstance(text, str):
        text = [text]

    try:
        results = await process_texts(text)
    except Exception as e:
        raise e

    return results


def chunk(
    text: Union[str, List[str]],
    chunk_size: int,
    token_counter: Callable[[str], int] = None,
    memoize: bool = True,
    verbose: bool = False,
) -> Union[List[str], List[List[str]]]:
    """Split a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

    Args:
        text (Union[str, List[str]]): The text to be chunked.
        chunk_size (int): The maximum number of tokens a chunk may contain.
        token_counter (Callable[[str], int]): A callable that takes a string and returns the number of tokens in it.
        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to `False`.

    Returns:
        Union[List[str], List[List[str]]]: A list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed.
    """
    import asyncio

    return asyncio.run(chunk_text(text, chunk_size, token_counter, memoize, verbose))


# ==================================================================================================
