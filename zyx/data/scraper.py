import requests
from typing import Union, List, Type, Optional
from pydantic import BaseModel
from ..completions.methods.extractor import extract
from ..resources.types import completion_create_params as params
from bs4 import BeautifulSoup
from .web_searcher import web_search


def limit_content_length(text: str, max_chars: int = 5000) -> str:
    """Limits the text content to a maximum number of characters."""
    return text[:max_chars]


def scrape(
    inputs: Union[str, List[str]],
    target: Optional[Type[BaseModel]] = None,
    model: Union[str, params.ChatModel] = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    process: str = "single",
    batch_size: int = 1, 
    search: bool = False,
    max_search_results: int = 10,
    max_chars_per_content: int = 5000,
    verbose: bool = False,
    **kwargs
):
    """
    Collects web content from given URLs or text inputs and extracts structured information.
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    if search:
        # Perform web search on each input query
        all_links = []
        for query in inputs:
            links = web_search(query, max_results=max_search_results, verbose=verbose)
            all_links.extend(links)
        inputs = all_links  # Use the links as inputs for scraping

    if not inputs:
        if verbose:
            print("No inputs to process after search.")
        return None

    contents = []
    for input_item in inputs:
        if input_item.startswith('http://') or input_item.startswith('https://'):
            # Fetch and process HTML content
            try:
                response = requests.get(input_item, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True)
                text_content = limit_content_length(text_content, max_chars=max_chars_per_content)
                contents.append(text_content)
                if verbose:
                    print(f"Extracted and limited text content from URL: {input_item}")
            except Exception as e:
                if verbose:
                    print(f"Error fetching URL: {input_item}, Error: {e}")
        else:
            # Assume input is text content
            text_content = limit_content_length(input_item, max_chars=max_chars_per_content)
            contents.append(text_content)
            if verbose:
                print(f"Added limited text content for processing: {input_item}")

    if not contents:
        if verbose:
            print("No content to process.")
        return None

    if target:
        # Adjust process to 'single' due to reduced batch size
        result = extract(
            target,
            contents,
            process='single',
            batch_size=batch_size,
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            verbose=verbose,
            **kwargs
        )
    else:
        # If no target model is provided, return the raw contents
        result = contents

    return result

if __name__ == "__main__":
    # Define a Pydantic model for structured data
    class Article(BaseModel):
        title: str
        content: str
        author: Optional[str] = None
        publication_date: Optional[str] = None

    # Example inputs: search queries
    inputs = [
        "latest technology news",
        "artificial intelligence advancements"
    ]

    # Extract structured data using the Article model and perform a web search
    results = scrape(
        inputs,
        target=Article,
        search=True,
        max_search_results=3,  # Reduced for testing
        model="gpt-4o-mini",
        process="single",  # Adjusted due to batch size
        batch_size=1,
        max_chars_per_content=5000,
        verbose=True
    )

    # Print the extracted results
    if results:
        if isinstance(results, list):
            for idx, result in enumerate(results, 1):
                print(f"\nResult {idx}:\n{result}")
        else:
            print(f"\nResult:\n{results}")
    else:
        print("No results were returned.")
