# Generating Synthetic Data with zyx BaseModel
# lets create some text data

import zyx as z

text = """
The two most commonly used attention functions are additive attention [2], and dot-product (multi-
plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of √1 . Additive attention computes the compatibility function using a feed-forward network with dk
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
"""

# lets create a model
class QuestionAnswerPair(z.BaseModel):
    question : str
    answer : str

# now we can generate using the model itself!
# all basemodel functions are prefixed with model_ to match the pydantic naming convention
response = QuestionAnswerPair.model_generate(
    instructions = f"Generate QA pairs from the following text: {text}",
    n = 5,
    model = "anthropic/claude-3-5-sonnet-latest"
)

print(response)