"""
scripts.run_readme_examples

Tests/validates all examples provided within the
readme.md file.

This also ensures that all examples are containerized in
terms of dependencies.
"""


def run_generate_content_example() -> None:
    import zyx

    result = zyx.make(
        target=int,
        context="What is 45+45?",
        model="openai:gpt-4o-mini",
    )

    print(result.output)


def run_parse_structured_data_example() -> None:
    import zyx
    from pydantic import BaseModel

    class Information(BaseModel):
        library_name: str
        library_description: str

    result = zyx.parse(
        source=zyx.paste("https://zyx.hammad.app"),
        target=Information,
        model="openai:gpt-4o-mini",
    )

    print(result.output.library_name)
    print(result.output.library_description)


def run_query_with_tools_example() -> None:
    import zyx
    from pydantic import BaseModel

    class Information(BaseModel):
        library_name: str
        library_description: str

    def log_website_url(url: str) -> None:
        print(f"Website URL: {url}")

    result = zyx.parse(
        source=zyx.paste("https://zyx.hammad.app"),
        target=Information,
        context=[
            {"role": "system", "content": "You are a web scraper."},
            # NOTE:
            # should test with a mock file at some point
            # zyx.paste("scraping_instructions.txt"),
            "[s]log the website URL before you parse.[/s]",
        ],
        model="openai:gpt-4o-mini",
        tools=[log_website_url],
    )

    print(result.output.library_name)
    print(result.output.library_description)


def run_edit_values_example() -> None:
    import zyx

    data = {"name": "John", "age": 30}

    result = zyx.edit(
        target=data,
        context="Update the age to 31",
        model="openai:gpt-4o-mini",
        merge=False
    )

    print(result.output)


def run_query_grounded_sources_example() -> None:
    import zyx

    result = zyx.query(
        source="Python is a high-level programming language...",
        target=str,
        context="What is Python?",
        model="openai:gpt-4o-mini",
    )

    print(result.output)


def run_select_from_options_example() -> None:
    import zyx
    from typing import Literal

    Color = Literal["red", "green", "blue"]

    result = zyx.select(
        target=Color,
        context="What color is the sky?",
        model="openai:gpt-4o-mini",
    )

    print(result.output)


def run_async_support_example() -> None:
    import asyncio

    async def test() -> None:
        import zyx

        result = await zyx.amake(
            target=str,
            context="Write a haiku about Python",
            model="openai:gpt-4o-mini",
        )

        print(result.output)

    asyncio.run(test())


def run_streaming_support_example() -> None:
    import zyx

    stream = zyx.make(
        target=str,
        context="Write a short story",
        model="openai:gpt-4o-mini",
        stream=True,
    )

    for chunk in stream.text(delta=True):
        print(chunk)


if __name__ == "__main__":
    # run_generate_content_example()
    # run_parse_structured_data_example()
    # run_query_with_tools_example()
    # run_edit_values_example()
    # run_query_grounded_sources_example()
    # run_select_from_options_example()
    # run_async_support_example()
    run_streaming_support_example()