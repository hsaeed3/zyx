from zyx import completion

response = completion(
    "Write a story about a man who travels through time to save his family from a deadly disease.",
    optimize = "costar"
)

print(response.choices[0].message.content)

