class LLMWrapper:
    """
    Abstract class for wrapping around the LLM.
    """

    total_tokens_used: int
    total_cost: int

    def __init__(self):
        self.total_tokens_used = 0
        self.total_cost = 0

    def make_request(self, message: str) -> str:
        """
        Makes a request to the LLM and return the response.

        :param message: an input to the LLM.
        :return: response from the LLM
        """
        raise NotImplementedError()

    def get_total_tokens_used(self) -> int:
        """
        Returns the total number of tokens used by the LLM.

        :return: total number of tokens used by the LLM.
        """
        return self.total_tokens_used

    def get_total_cost(self) -> int:
        """
        Returns the total cost of the LLM so far.

        :return: the total cost of the LLM so far.
        """
        return self.total_cost
