# >>>>>>>>>>>>>>>>>>>>>>>>>>>
# zyx is open source
# use it however you want :)
#
# 2024 Hammad Saeed
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<

from setuptools import setup, find_packages

setup(
    name="zyx",
    version="0.2",
    author="Hammad Saeed",
    author_email="hammad@supportvectors.com",

    description="Lightspeed Python functions for the AI era.",

    python_requires=">3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[

        # Foundational / Utility Libraries
        "art", "fastui", "loguru", "pathlib", "tqdm",

        # Data / RAG
        "chroma", "peewee", "sqlmodel",

        # LLMs
        "litellm", "ollama",

        # Prompting
        "dspy-ai", "instructor",

        # API / Web
        "fastapi", "uvicorn"

])
