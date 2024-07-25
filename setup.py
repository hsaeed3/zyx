# ==============================================================================
# ~ zyx ~
# hammad saeed & other people with their stuff
# nothing fancy, just a toolkit of other people's stuff
# ==============================================================================

# ==============================================================================
from setuptools import find_packages, setup
# ==============================================================================

# ==============================================================================
setup(
    name="zyx",
    version="0.1.95",
    # --------------------------------------------------------------------------
    author="Hammad Saeed",
    author_email="hvmmad@gmail.com",
    # --------------------------------------------------------------------------
    description="A fun ai toolkit of other people's stuff",
    # --------------------------------------------------------------------------
    python_requires=">=3.9",
    # --------------------------------------------------------------------------
    packages=find_packages("src"),
    package_dir={"": "src"},
    # --------------------------------------------------------------------------
    install_requires=[
        # CORE DEPENDENCIES ====================================================
        # -- genai -------------------------------------------------------------
        "instructor",
        "cohere",  # Cohere will be removed on fix
        "litellm",
        "huggingface_hub",
        # -- data --------------------------------------------------------------
        "langchain-core",
        "langchain-openai",
        "langchain-anthropic",
        "mem0ai",
        "langgraph",
        "sqlmodel",
        "qdrant-client",
        # -- util --------------------------------------------------------------
        "fastapi",
        "loguru",
        "rich",
        "tqdm",
        "uvicorn",
        # =======================================================================
    ],
    # --------------------------------------------------------------------------
    extras_require = {
        "kernel" : [
            "semantic-kernel"
        ]
    }
)
# ==============================================================================
