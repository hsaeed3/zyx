# zyx.completions
# base client for llm completions


class Completions:

    """Base Completions Resource"""

    # init
    # no client args in init anymore, litellm.completion
    # is the default client/completion method; no class
    # is instantiated anymore
    def __init__(
            self,
            # verbosity
            verbose : bool = False
    ):

        # set config
        self.verbose = verbose

    