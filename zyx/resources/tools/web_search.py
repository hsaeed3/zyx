from typing import Optional, List, Dict, Any


def web_search(query: str, max_results: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    A tool that searches the web for information using DuckDuckGo Search API.

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return (Default: 5)

    Returns:
        - A list of dictionaries containing 'title' and 'href' of the search results
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print(
            "duckduckgo_search is not installed, please install it with `pip install duckduckgo_search`"
        )
        raise ImportError

    results = DDGS().text(keywords=query, max_results=max_results)
    return results


if __name__ == "__main__":
    print(web_search("What is the weather in Tokyo?"))
