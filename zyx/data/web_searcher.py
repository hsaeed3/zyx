from typing import List, Optional
from duckduckgo_search import DDGS


def web_search_tool(
        query : str,
        max_results : Optional[int] = 10,
        verbose : bool = False
) -> List[str]:
    """Searches the web for a given query"""
    return web_search(query, max_results, verbose)


def web_search(query: str, max_results: int = 10, verbose: bool = False) -> List[str]:
    """
    Performs a web search using DuckDuckGo and returns the links of the search results.

    Example:
    ```python
    links = web_search("latest technology news")
    ```

    Returns:
        List[str]: A list of links to the search results.
    """
    client = DDGS()

    if verbose:
        print(f"Performing web search for query: {query}")

    results = client.text(query, max_results=max_results)
    if not results:
        if verbose:
            print("No search results found.")
        return []

    links = [result['href'] for result in results if 'href' in result]
    if verbose:
        print(f"Found {len(links)} links from search results.")
    return links