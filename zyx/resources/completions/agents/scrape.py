try:
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from typing import List, Optional, Dict, Any, Literal
    from pydantic import BaseModel, Field
    from enum import Enum

    from ....client import Client, InstructorMode, ToolType
    from ....resources.completions.base.generate import generate
    from ....lib.types.document import Document
    from ....lib.utils.logger import get_logger

    import warnings
except ImportError:
    import os

    print(
        "The [bold]`zyx(data)`[/bold] data extension is required to use this module. Install it?"
    )
    if input("Install? (y/n)") == "y":
        os.system("pip install 'zyx[data]'")
    else:
        print("Exiting...")
        exit(1)


logger = get_logger("scrape")


def web_search(query: str, max_results: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    A tool that searches the web for information using DuckDuckGo Search API.

    Parameters:
        query: The query to search for
        max_results: The maximum number of results to return (Default: 5)

    Returns:
        A list of dictionaries containing 'title' and 'href' of the search results
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


class ScrapingStep(Enum):
    SEARCH = "search"
    FETCH = "fetch"
    SUMMARIZE = "summarize"
    EVALUATE = "evaluate"
    REFINE = "refine"


class StepResult(BaseModel):
    is_successful: bool
    explanation: str
    content: Optional[str] = None


class ScrapeWorkflow(BaseModel):
    query: str
    current_step: ScrapingStep = ScrapingStep.SEARCH
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    fetched_contents: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    evaluation: Optional[StepResult] = None


class QueryList(BaseModel):
    queries: List[str]


def scrape(
    query: str,  # Single query input
    num_queries: int = 5,  # Number of queries to generate
    max_results: Optional[int] = 5,
    workers: int = 5,
    model: str = "gpt-4o-mini",
    client: Literal["openai", "litellm"] = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: InstructorMode = "tool_call",
    response_model: Optional[BaseModel] = None,
    max_retries: int = 3,
    temperature: float = 0.5,
    run_tools: Optional[bool] = False,
    tools: Optional[List[ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    verbose: Optional[bool] = False,
    **kwargs,
) -> Document:
    """
    Scrapes the web for topics & content about multiple queries, generates a well-written summary, and returns a Document object.

    Parameters:
        query: The initial search query.
        num_queries: Number of queries to generate based on the initial query.
        max_results: Maximum number of search results to process.
        workers: Number of worker threads to use.
        model: The model to use for completion.
        client: The client to use for completion.
        api_key: The API key to use for completion.
        base_url: The base URL to use for completion.
        mode: The mode to use for completion.
        max_retries: The maximum number of retries to use for completion.
        temperature: The temperature to use for completion.
        run_tools: Whether to run tools for completion.
        tools: The tools to use for completion.

    Returns:
        A Document object containing the summary and metadata.
    """

    warnings.warn(
        "The scrape function will no longer be updated."
        "Go to https://github.com/unclecode/crawl4ai for an incredibly robust & feature-rich web-scraping tool.",
        DeprecationWarning,
    )

    import threading
    from bs4 import BeautifulSoup

    completion_client = Client(
        api_key=api_key, base_url=base_url, provider=client, verbose=verbose
    )

    # Generate multiple queries based on the initial query
    query_list = generate(
        target=QueryList,
        instructions=f"Generate {num_queries} related search queries based on the initial query: '{query}'",
        n=1,
        model=model,
        api_key=api_key,
        base_url=base_url,
        client=client,
        verbose=verbose,
    ).queries

    workflow = ScrapeWorkflow(query=query)  # Use the initial query for workflow

    if verbose:
        print(f"Starting scrape for queries: {query_list}")

    all_search_results = []
    all_urls = []

    for query in query_list:
        # Step 1: Use web_search() to get search results
        workflow.current_step = ScrapingStep.SEARCH
        search_results = web_search(query, max_results=max_results)
        all_search_results.extend(search_results)
        urls = [result["href"] for result in search_results if "href" in result]
        all_urls.extend(urls)

        if verbose:
            print(f"Found {len(urls)} URLs for query: {query}")

    workflow.search_results = all_search_results

    # Step 2: Define a function to fetch and parse content from a URL
    def fetch_content(url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            texts = soup.find_all(text=True)
            visible_texts = filter(tag_visible, texts)
            content = " ".join(t.strip() for t in visible_texts)
            return content
        except Exception as e:
            if verbose:
                print(f"Error fetching {url}: {e}")
            return ""

    # Helper function to filter visible text
    from bs4.element import Comment

    def tag_visible(element):
        if element.parent.name in [
            "style",
            "script",
            "head",
            "title",
            "meta",
            "[document]",
        ]:
            return False
        if isinstance(element, Comment):
            return False
        return True

    # Step 3: Use ThreadPoolExecutor to fetch content in parallel
    workflow.current_step = ScrapingStep.FETCH
    contents = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {executor.submit(fetch_content, url): url for url in all_urls}
        for future in future_to_url:
            content = future.result()
            if content:
                contents.append(content)

    workflow.fetched_contents = contents

    if verbose:
        print(f"Collected content from {len(contents)} pages")

    # Step 4: Combine the content
    workflow.current_step = ScrapingStep.SUMMARIZE
    combined_content = "\n\n".join(contents)

    # Step 4.5: If Response Model is provided, return straight away
    if response_model:
        return completion_client.completion(
            messages=[
                {"role": "user", "content": "What is our current scraped content?"},
                {"role": "assistant", "content": combined_content},
                {
                    "role": "user",
                    "content": "Only extract the proper content from the response & append into the response model.",
                },
            ],
            model=model,
            response_model=response_model,
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

    # Step 5: Use the completion function to generate a summary
    # Prepare the prompt
    system_prompt = (
        "You are an AI assistant that summarizes information gathered from multiple web pages. "
        "Ensure that all links parsed are from reputable sources and do not infringe any issues. "
        "Provide a comprehensive, well-written summary of the key points related to the following query."
    )
    user_prompt = f"Query: {query}\n\nContent:\n{combined_content}\n\nSummary:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call the completion function
    response = completion_client.completion(
        messages=messages,
        model=model,
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
    summary = response.choices[0].message.content
    workflow.summary = summary

    # Step 6: Evaluate
    workflow.current_step = ScrapingStep.EVALUATE
    evaluation_prompt = (
        f"Evaluate the quality and relevance of the following summary for the query: '{query}'\n\n"
        f"Summary:\n{summary}\n\n"
        "Ensure that all links parsed are from reputable sources and do not infringe any issues. "
        "Provide an explanation of your evaluation and determine if the summary is successful or needs refinement."
    )

    evaluation_response = completion_client.completion(
        messages=[
            {"role": "system", "content": "You are an expert evaluator of summaries."},
            {"role": "user", "content": evaluation_prompt},
        ],
        model=model,
        response_model=StepResult,
        mode=mode,
        max_retries=max_retries,
        temperature=temperature,
    )

    workflow.evaluation = evaluation_response

    # Step 7: Refine if necessary
    if not evaluation_response.is_successful:
        workflow.current_step = ScrapingStep.REFINE
        refine_prompt = (
            f"The previous summary for the query '{query}' needs improvement.\n\n"
            f"Original summary:\n{summary}\n\n"
            f"Evaluation feedback:\n{evaluation_response.explanation}\n\n"
            "Ensure that all links parsed are from reputable sources and do not infringe any issues. "
            "Please provide an improved and refined summary addressing the feedback."
        )

        refined_response = completion_client.completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refining and improving summaries.",
                },
                {"role": "user", "content": refine_prompt},
            ],
            model=model,
            mode=mode,
            max_retries=max_retries,
            temperature=temperature,
        )

        summary = refined_response.choices[0].message.content

    if verbose:
        print("Generated summary:")
        print(summary)

    # Create a Document object
    document = Document(
        content=summary,
        metadata={
            "query": query,
            "urls": all_urls,
            "model": model,
            "client": client,
            "workflow": workflow.model_dump(),
        },
    )

    return document


# Example usage
if __name__ == "__main__":
    result_document = scrape(
        query="Latest advancements in renewable energy",
        num_queries=5,
        max_results=5,
        workers=5,
        verbose=True,
    )
    print("Final Document:")
    print(result_document.content)

    class YoutubeLinks(BaseModel):
        links: list[str] = Field(description="A list of youtube links")

    result_document = scrape(
        query="Latest advancements in renewable energy",
        num_queries=5,
        max_results=5,
        workers=5,
        verbose=True,
        response_model=YoutubeLinks,
    )
