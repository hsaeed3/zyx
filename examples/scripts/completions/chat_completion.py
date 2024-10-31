# Standard Chat Completion

import zyx as z

# simple example
# get a response
response = z.completion(
    "What is the capital of France?"
)

# this returns a `ChatCompletion` object
print("ChatCompletion: \n\n" + str(response) + "\n\n")

# extended example
# get a response with a structured output
response = z.completion(
    messages = [
        {"role" : "system", "content" : "You only speak in spanish."},
        {"role" : "user", "content" : "What is the capital of France?"}
    ],
    # any litellm model is supported
    model = "gpt-4o-mini",
)

# lets print the response content
print("Response Content: \n\n" + str(response.choices[0].message.content) + "\n\n")
