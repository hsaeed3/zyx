import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Literal
from ...types import client as clienttypes
from ...types.document import Document


def web_search(query: str, max_results: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    A tool that searches the web for information using DuckDuckGo Search API.

    Parameters:
        - query: The query to search for
        - max_results: The maximum number of results to return (Default: 5)

    Returns:
        - A list of dictionaries containing 'title' and 'href' of the search results
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("duckduckgo_search is not installed, please install it with `pip install duckduckgo_search`")
        raise ImportError

    results = DDGS.text(keywords = query, max_results=max_results)
    return results

def scrape(
    query: str,
    max_results: Optional[int] = 5,
    workers: int = 5,
    model: str = "gpt-4o-mini",
    client: Literal["openai", "litellm"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: clienttypes.InstructorMode = "chat",
    max_retries: int = 3,
    temperature: float = 0.5,
    run_tools: Optional[bool] = False,
    tools: Optional[List[clienttypes.ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    verbose: Optional[bool] = False,
    **kwargs,
) -> Document:
    """
    Scrapes the web for topics & content about a query, generates a well-written summary, and returns a Document object.

    Parameters:
        - query: The search query.
        - max_results: Maximum number of search results to process.
        - workers: Number of worker threads to use.
        - Other parameters are passed to the completion function.

    Returns:
        - A Document object containing the summary and metadata.
    """
    import threading
    from ..client import completion
    from bs4 import BeautifulSoup

    if verbose:
        print(f"Starting scrape for query: {query}")

    # Step 1: Use web_search() to get search results
    search_results = web_search(query, max_results=max_results)
    urls = [result['href'] for result in search_results if 'href' in result]

    if verbose:
        print(f"Found {len(urls)} URLs")

    # Step 2: Define a function to fetch and parse content from a URL
    def fetch_content(url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text content from the page
            # You might want to refine this to get more meaningful content
            texts = soup.find_all(text=True)
            visible_texts = filter(tag_visible, texts)
            content = u" ".join(t.strip() for t in visible_texts)
            return content
        except Exception as e:
            if verbose:
                print(f"Error fetching {url}: {e}")
            return ""

    # Helper function to filter visible text
    from bs4.element import Comment
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    # Step 3: Use ThreadPoolExecutor to fetch content in parallel
    contents = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {executor.submit(fetch_content, url): url for url in urls}
        for future in future_to_url:
            content = future.result()
            if content:
                contents.append(content)

    if verbose:
        print(f"Collected content from {len(contents)} pages")

    # Step 4: Combine the content
    combined_content = "\n\n".join(contents)

    # Optionally, you can chunk the content if it's too large
    # For now, we'll assume it's manageable

    # Step 5: Use the completion function to generate a summary
    # Prepare the prompt
    system_prompt = (
        "You are an AI assistant that summarizes information gathered from multiple web pages. "
        "Provide a comprehensive, well-written summary of the key points related to the following query."
    )
    user_prompt = f"Query: {query}\n\nContent:\n{combined_content}\n\nSummary:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call the completion function
    response = completion(
        messages=messages,
        model=model,
        client=client,
        api_key=api_key,
        base_url=base_url,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
        run_tools=run_tools,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        verbose=verbose,
        **kwargs,
    )

    # Extract the summary
    summary = response.choices[0].message["content"]

    if verbose:
        print("Generated summary:")
        print(summary)

    # Create a Document object
    document = Document(
        content=summary,
        metadata={
            "query": query,
            "urls": urls,
            "model": model,
            "client": client,
        }
    )

    return document

# Example usage
if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = 'YOUR_API_KEY'
    result_document = scrape(
        query="Latest advancements in renewable energy",
        max_results=5,
        workers=5,
        api_key=api_key,
        verbose=True,
    )
    print("Final Document:")
    print(result_document.content)
