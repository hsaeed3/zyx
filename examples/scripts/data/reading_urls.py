# Reading URLs with zyx

import zyx as z

# simple url read task
response = z.data.read_url(
    "https://python-client.qdrant.tech/quickstart"
)

print(response)