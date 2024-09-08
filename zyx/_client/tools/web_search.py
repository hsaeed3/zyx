from typing import Optional, Union


def web_search(query: str, max_results: Optional[int] = 5) -> Union[dict, bool]:
    """A tool that searches the web for information

    Parameters:
        - query: The query to search for
        - max_results: The maximum number of results to return (Default: 5)
    """
    from tavily import TavilyClient
    import os
    
    key = os.getenv("TAVILY_API_KEY")

    if not key:
        raise ValueError(
            "TAVILY_API_KEY is not set, please set the environment variable"
        )

    client = TavilyClient(api_key=key)

    try:
        return client.search(
            query=query, search_depth="advanced", max_results=max_results
        )
    except Exception as e:
        return False


if __name__ == "__main__":
    print(web_search("What is the weather in Tokyo?"))
