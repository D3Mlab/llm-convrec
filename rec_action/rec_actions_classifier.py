from rec_action.rec_action import RecAction
from state.state_manager import StateManager


class RecActionsClassifier:
    """
    Abstract class responsible for classifying recommender action for the recommender's response to the most recent
    user's input.

    :param rec_actions: all possible recommender actions
    """

    _rec_actions: list[RecAction]

    def __init__(self, rec_actions: list[RecAction]):
        self._rec_actions = rec_actions

    def classify(self, state_manager: StateManager, k: int = 1) -> list[RecAction]:
        """
        Returns k recommender actions in self._rec_responses for responding to the most recent user's input.

        :param state_manager: current state representing the conversation
        :param k: size of the list returned
        :return: recommender actions appropriate for responding to the most recent user's input or None if there are no
                 appropriate recommender action
        """
        raise NotImplementedError()

    def get_rec_actions(self) -> list[RecAction]:
        """
        Returns all possible recommender actions for this classifier.

        :return: all possible recommender actions for this classifier.
        """
        return self._rec_actions.copy()
