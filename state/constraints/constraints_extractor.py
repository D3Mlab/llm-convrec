from state.message import Message


class ConstraintsExtractor:
    """
    Abstract class used to extract constraints from the most recent user's input.

    :param default_keys: set of possible keys for the constraints
    """
    _default_keys: list[str]

    def __init__(self, default_keys: list[str] = None) -> None:
        if default_keys is None:
            default_keys = ["location"]
        self._default_keys = default_keys

    def extract(self, conv_history: list[Message]) -> dict:
        """
        Extract the constraints from the most recent user's input, and return them.

        :param conv_history: current conversation history
        :return: constraints that was updated in this function.
        """
        raise NotImplementedError()

