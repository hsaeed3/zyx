import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    app_title="Examples: Make",
    css_file="",
    html_head_file="../head.html",
)


@app.cell
def _():
    import marimo as mo
    from rich import print

    return mo, print


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Semantic Operations: **Make**

    The `make()` semantic operation is the most straightforward generative function within the `ZYX` library. It's main (and only purpose) is to generate an output based on a given **`target`** type or value.

    Before we get started with the first example, go ahead and configure the AI model you'd like to use for running the following blocks. In all semantic operation notebooks, a model should be given in the `pydantic_ai` format, which is represented as *provider:model*, for example: *openai:gpt-4o-mini*
    """)
    return


@app.cell
def _(mo):
    model = mo.ui.text(
        label="Model: ",
        placeholder="openai:gpt-4o-mini",
        value="openai:gpt-4o-mini",
    )
    model
    return (model,)


@app.cell
def _(model):
    model_name = model.value if model.value else "openai:gpt-4o-mini"
    return (model_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using **Make** To Generate Structured Outputs

    Let's begin with our first example of the `make()` function, generating a simple structured output in a `target` type. In this case we'll generate a number using `target=int`.
    """)
    return


@app.cell
def _(mo, model_name):
    # import 'make'
    from zyx import make

    # generate a numerical response
    result = make(
        # the 'target' type controls the output of the make operation, as well as most other
        # semantic operations
        target=int,
        # `context` is the universal parameter or entrypoint for passing prompts/message history and more
        # to your models and agents
        context="What is 45+45-20?",
        # all `pydantic_ai` models are compatible with zyx
        model=model_name,
    )
    mo.show_code()
    return make, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's take a look at our output. All semantic operations, return either the **`Result`** or **`Stream`** objects from `zyx/result.py` or `zyx.stream.py` based on if `stream` is True or False.
    """)
    return


@app.cell
def _(print, result):
    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can access the final output of a semantic operation by accessing `Result.output`.
    """)
    return


@app.cell
def _(print, result):
    print(result.output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Generating Synthetic Data

    A quick and easy way to generate synthetic data through the `make` semantic operation, is by not passing `context` at all, this allows the model to generate diverse outputs for a given target. Lets create a slightly more complex example now.
    """)
    return


@app.cell
def _(make, mo, model_name):
    from dataclasses import dataclass

    @dataclass
    class User:
        name: str
        age: int
        address: str

    # as you can see here we havent passed any instructions or context at all.
    user = make(User, model=model_name)
    mo.show_code()
    return (user,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now we can view the output of our generation.
    """)
    return


@app.cell
def _(print, user):
    print(user)
    return


if __name__ == "__main__":
    app.run()
