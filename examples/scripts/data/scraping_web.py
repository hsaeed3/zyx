# Scraping Web Data with zyx

import zyx as z

# simple web scrape task
response = z.data.scrape(
    "The top research papers in AI for October 2024",
    search = True
)

# print the response
# this will return a bunch of unstructured html text
print(response)