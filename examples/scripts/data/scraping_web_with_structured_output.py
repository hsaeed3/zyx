# Scraping Web Data with zyx

import zyx as z

class BestPapers(z.BaseModel):
    paper_names : list[str]

# simple web scrape task
response = z.data.scrape(
    "The top ARXIV research papers in AI for October 2024",
    # give a target model to extract structured information
    target = BestPapers,
    # ability to limit the number of search results
    max_search_results = 2,
    search = True,
)

# print the response
print(response)