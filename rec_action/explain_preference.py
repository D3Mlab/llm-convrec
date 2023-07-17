from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from state.message import Message


class ExplainPreference(RecAction):
    """
    Class representing Explain Preference recommender action.
    """

    def __init__(self, priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)

    def get_name(self):
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Explain Preference"

    def get_description(self):
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender explains recommendations based on the user's said preference"

    def get_prompt_response(self, state_manager: StateManager) -> str | None:
        """
        Return prompt based recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: prompt based recommender's response corresponding to this action
        """
        return f"Give me an explanation of why you made this recommendation based on this state, but do not " \
            f"mention the state {str(state_manager)}"

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        return None

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.
        :return: whether hard coded response exists or not.
        """
        return False

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        # TODO: Implement this part
        return self.priority_score_range[0] - 1

    def update_state(self, state_manager: StateManager, response: str, **kwargs):
        """
        Updates the state based off of recommenders response

        :param state_manager: current state representing the conversation
        :param response: recommender response msg that is returned to the user
        :param **kwargs: misc. arguments 

        :return: none
        """

        message = Message("recommender", response)
        state_manager.update_conv_history(message)