import semchunk
import tiktoken
from typing import Union, List
from pydantic import BaseModel

from ..types.document import Document


def chunk(
    inputs: Union[str, Document, List[Union[str, Document]]], 
    chunk_size: int = 512, 
    model: str = "gpt-4",
    processes: int = 1, 
    memoize: bool = True, 
    progress: bool = False, 
    max_token_chars: int = None 
) -> Union[List[str], List[List[str]]]:
    """
    Takes a string, Document, or a list of strings/Document models and returns the chunked content.
    """
    try:
        tokenizer = tiktoken.encoding_for_model(model)
        chunker = semchunk.chunkerify(
            tokenizer, 
            chunk_size=chunk_size, 
            max_token_chars=max_token_chars, 
            memoize=memoize
        )

        # Handle single input case (str or Document)
        if isinstance(inputs, (str, Document)):
            inputs = [inputs]  # Convert to list for uniform handling

        if not isinstance(inputs, list):
            raise TypeError("inputs must be a string, Document, or a list of strings/Documents")

        texts = []
        for item in inputs:
            # Handle Document content
            if isinstance(item, Document):
                content = item.content
                # Convert non-string content (e.g., lists from CSV/XLSX) to string
                if isinstance(content, list):
                    content = "\n".join([" | ".join(map(str, row)) for row in content])
                elif not isinstance(content, str):
                    raise TypeError(f"Document content must be a string or list of strings, found {type(content)}")
                texts.append(content)
            # Handle string input directly
            elif isinstance(item, str):
                texts.append(item)
            else:
                raise TypeError(f"Unsupported input type: {type(item)}")

        # Chunk the content, using processes and progress bar as needed
        if len(texts) == 1:
            return chunker(texts[0])  # Single input, return the chunked result
        else:
            return chunker(texts, processes=processes, progress=progress)  # Multiple inputs

    except Exception as e:
        # Detailed error logging
        print(f"Error in chunk function: {str(e)}")
        raise e
