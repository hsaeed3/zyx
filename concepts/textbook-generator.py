"""
Textbook Generator

zyx examples - i

This example showcases a quick method to generate a 'textbook' using web-scraped information.
"""

# This example will individually import all the necessary modules from zyx
from zyx import (
    scrape,  # For generating our web research
    BaseModel,
    Field,  # For generating our textbook & model definitions
)


def generate_textbook(
    instructions: str,
    model: str = "gpt-4o-mini",
) -> BaseModel:
    # Lets first collect our information
    ai_research = scrape(
        instructions,
        model="gpt-4o-mini",
        num_queries=5,  # This determines how many queries are generated based on our original query
        workers=5,  # This determines how many threads are used for our web search
        max_results=5,  # This determines the maximum number of results we scrape from the web
        verbose=True,  # This determines if we want to print out the raw results as we scrape them
    )

    # Now that we have our research; lets generate our textbook
    # First, we'll have to define our textbook using Pydantic
    class TextbookChapter(BaseModel):
        title: str = Field(description="The title of the chapter")
        content: str = Field(description="The content of the chapter")

    class Textbook(BaseModel):
        title: str = Field(description="The title of the textbook")
        chapters: list[TextbookChapter] = Field(
            description="The chapters of the textbook"
        )

    textbook = Textbook.generate(
        instructions=f"""
        We have collecting the following information.

        <research>
        {ai_research}
        </research>

        Please use this information to create a comprehensive textbook on {instructions}.
        """,
        model="gpt-4o-mini",
        process="sequential",  # This generates each field in the model one by one
        verbose=True,
    )

    # Lets write our textbook to a markdown file
    # We will have to create it if it doesnt exist
    with open("textbook.md", "w") as f:
        f.write(textbook.model_dump_json())

    return textbook


if __name__ == "__main__":
    textbook = generate_textbook(
        instructions="AI & LLMs in September 2024", model="gpt-4o-mini"
    )

    print(textbook)
