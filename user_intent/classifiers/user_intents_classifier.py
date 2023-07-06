from state.state_manager import StateManager
from user_intent.user_intent import UserIntent


class UserIntentsClassifier:
    """
    Abstract class responsible for classifying most recent user's input to one of
    the user intent in the given list.

    :param user_intents: all possible user intents
    """

    _user_intents: list[UserIntent]

    def __init__(self, user_intents: list[UserIntent]):
        self._user_intents = user_intents

    def classify(self, curr_state: StateManager) -> list[UserIntent]:
        """
        Returns one of the user intent in self._user_intents corresponding to the most recent user's input in the
        given state.
        Returns None if none of user intents in self._user_intents corresponds to the most recent user's input.

        :param curr_state: current state representing the conversation
        :return: user intent corresponding to the most recent user's input or None if there are no user intent
                 corresponding to the user's input
        """
        raise NotImplementedError()

    def get_user_intents(self) -> list[UserIntent]:
        """
        Returns all possible user intents for this classifier.

        :return: all possible user intents for this classifier.
        """
        return self._user_intents.copy()
