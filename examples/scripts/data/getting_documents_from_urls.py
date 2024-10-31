# Getting Documents from URLs with zyx

# the read() function is able to read URL's specifically
# if the URL links to a document or type of document file.

import zyx as z

# simple url read task
# lets read a paper from arxiv

paper = z.data.read(
    "https://arxiv.org/pdf/2410.18933.pdf"
)

print(paper)