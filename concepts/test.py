from zyx import data, llm

# Load the document (Located in the documents/ directory)
document = data.read("documents/large_language_monkeys.pdf")

chunks = data.chunk(document)