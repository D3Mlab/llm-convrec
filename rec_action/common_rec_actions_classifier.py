from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from rec_action.rec_actions_classifier import RecActionsClassifier


class CommonRecActionsClassifier(RecActionsClassifier):
    """
    Implementation for RecActionsClassifier that classifies recommender action using conditionals.

    :param rec_actions: all possible recommender actions
    """

    _rec_actions: list[RecAction]

    def __init__(self, rec_actions: list[RecAction]):
        super().__init__(rec_actions)

    def classify(self, state_manager: StateManager, k: int = 1) -> list[RecAction]:
        """
        Returns top k recommender action in self._rec_responses for responding to the most recent user's input.
        Returns None if none of recommender actions in self._user_intents is appropriate.

        It returns the first RecAction in self._rec_responses where .get_priority_score returns True.

        :param state_manager: current state representing the conversation
        :param k: length of the list returned
        :return: recommender action appropriate for responding to the most recent user's input or None if there are no
                 appropriate recommender action
        """
        result = sorted(self._rec_actions, key=lambda x: x.get_priority_score(
            state_manager), reverse=True)[:k]

        for i in range(len(result)):
            if result[i].get_priority_score(state_manager) < result[i].priority_score_range[0]:
                return result[:i]
        return result
