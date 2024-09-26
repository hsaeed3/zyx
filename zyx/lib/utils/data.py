import os
import hashlib
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Callable, Any, Union
import re
import json
import csv

from ..types.document import Document


def load_docs(file_paths: List[Path], loader_func: Callable[[Path], str]) -> List[str]:
    """
    Load documents in parallel using a provided loader function.
    
    Args:
    file_paths (List[Path]): List of file paths to load.
    loader_func (Callable[[Path], str]): Function to load a single document.
    
    Returns:
    List[str]: List of loaded documents.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        documents = list(executor.map(loader_func, file_paths))
    return documents


def simple_text_loader(file_path: Path) -> str:
    """
    Simple loader for text files.
    
    Args:
    file_path (Path): Path to the text file.
    
    Returns:
    str: Loaded document content.
    """
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    return content


def extract_metadata(file_path: Path) -> Dict[str, str]:
    """
    Extract metadata from a filename.
    
    Args:
    file_path (Path): The file path to extract metadata from.
    
    Returns:
    Dict[str, str]: Extracted metadata.
    """
    name = file_path.stem
    parts = re.split(r'[-_]', name)
    metadata = {
        'file_name': file_path.name,
        'file_type': file_path.suffix[1:],  # Remove the leading dot
    }
    if len(parts) > 1:
        metadata['title'] = ' '.join(parts[:-1])
        metadata['date'] = parts[-1]
    else:
        metadata['title'] = name
    return metadata


def hash_documents(docs: Union[str, Document, List[Union[str, Document]]]) -> List[str]:
    """
    Calculate MD5 hashes for a list of document contents in parallel.
    
    Args:
    docs (Union[str, Document, List[Union[str, Document]]]): List of document contents to hash.
    
    Returns:
    List[str]: List of MD5 hashes.
    """
    def calculate_hash(content: str) -> str:
        hasher = hashlib.md5()
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()

    # Handle single string or Document by converting to list
    if isinstance(docs, (str, Document)):
        docs = [docs]

    contents = [convert_to_string(doc) for doc in docs]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        hashes = list(executor.map(calculate_hash, contents))

    return hashes


def extract_keywords(doc: Union[str, Document, List[Union[str, Document]]], top_n: int = 10) -> List[str]:
    """
    Extract top keywords from a document using TF-IDF.
    Note: This is a simplified version. For production, consider using libraries like sklearn.
    
    Args:
    doc (Union[str, Document, List[Union[str, Document]]]): The document content to extract keywords from.
    top_n (int): Number of top keywords to extract.
    
    Returns:
    List[str]: List of extracted keywords.
    """
    # Handle single string or Document by converting to list
    if isinstance(doc, (str, Document)):
        doc = [doc]

    content = convert_to_string(doc[0])
    words = re.findall(r'\w+', content.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    tfidf = {word: freq / len(words) for word, freq in word_freq.items()}
    top_keywords = sorted(tfidf, key=tfidf.get, reverse=True)[:top_n]
    
    return top_keywords


def summarize(doc: Union[str, Document, List[Union[str, Document]]], summary_length: int = 200, model: str = "gpt-3.5-turbo") -> str:
    """
    Create a summary of the document using a language model.
    
    Args:
    doc (Union[str, Document, List[Union[str, Document]]]): The document content to summarize.
    summary_length (int): Approximate length of the summary in characters.
    model (str): The model to use for completion.
    
    Returns:
    str: The summary of the document.
    """
    from ..completions.client import completion

    # Handle single string or Document by converting to list
    if isinstance(doc, (str, Document)):
        doc = [doc]

    content = convert_to_string(doc[0])
    prompt = f"""Please provide a concise summary of the following text in approximately {summary_length} characters:

{content}

Summary:"""

    try:
        response = completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=summary_length // 4
        )
        
        if isinstance(response, str):
            summary = response.strip()
        elif isinstance(response, dict) and 'content' in response:
            summary = response['content'].strip()
        else:
            raise ValueError("Unexpected response format from completion function")

    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = content[:summary_length] + "..."

    return summary


def export_documents_to_json(docs: Union[str, Document, List[Union[str, Document]]], metadata_list: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Export a list of document contents and metadata to a JSON file.
    
    Args:
    docs (Union[str, Document, List[Union[str, Document]]]): List of document contents to export.
    metadata_list (List[Dict[str, Any]]): List of metadata dictionaries corresponding to the contents.
    output_file (Path): Path to the output JSON file.
    """
    # Handle single string or Document by converting to list
    if isinstance(docs, (str, Document)):
        docs = [docs]

    contents = [convert_to_string(doc) for doc in docs]
    data = [{'content': content, 'metadata': metadata} for content, metadata in zip(contents, metadata_list)]
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def import_documents_from_json(input_file: Path) -> List[Dict[str, Any]]:
    """
    Import documents from a JSON file.
    
    Args:
    input_file (Path): Path to the input JSON file.
    
    Returns:
    List[Dict[str, Any]]: List of imported document contents and metadata.
    """
    with input_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_document_report(metadata_list: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Generate a CSV report of document metadata.
    
    Args:
    metadata_list (List[Dict[str, Any]]): List of metadata dictionaries to report on.
    output_file (Path): Path to the output CSV file.
    """
    fieldnames = set()
    for metadata in metadata_list:
        fieldnames.update(metadata.keys())
    
    with output_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for metadata in metadata_list:
            writer.writerow(metadata)


def convert_to_string(doc: Union[str, Document]) -> str:
    """
    Convert Document or a string to a string.
    
    Args:
    doc (Union[str, Document]): The document to convert.
    
    Returns:
    str: The document content as a string.
    """
    if isinstance(doc, Document):
        return doc.content
    return doc


def convert_to_document(doc: Union[str, Document]) -> Document:
    """
    Convert a string to a Document.
    
    Args:
    doc (Union[str, Document]): The input document.
    
    Returns:
    Document: A Document instance.
    """
    if isinstance(doc, Document):
        return doc
    return Document(content=doc, metadata={"file_type": "text"})


if __name__ == "__main__":
    # Example usage of the functions
    text = """
    John is a software engineer. He works for a tech company.
    Some people say that John is a genius.
    Apparently, John has been sneaking around the office at night.
    That is not the kind of behavior we expect from our employees.
    """

    # Example: Hashing documents
    document = Document(content=text, metadata={"title": "Sample Document"})
    hashes = hash_documents([document])
    print(f"MD5 Hashes: {hashes}")

    # Example: Extracting keywords
    keywords = extract_keywords(document)
    print(f"Keywords: {keywords}")

    # Example: Summarizing
    summary = summarize(document)
    print(f"Summary: {summary}")