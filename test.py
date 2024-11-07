from zyx.completions import Completions


# initialize completions client
completions = Completions(verbose=True)


print(completions.completion(
    "hi", response_model = "response"))