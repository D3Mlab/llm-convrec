from textwrap import dedent

from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.accept_recommendation import AcceptRecommendation
from state.message import Message


class PostAcceptanceAction(RecAction):
    """
    Class representing Answer recommender action.
    """

    def __init__(self, hard_coded_responses: list[dict], priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)
        self._hard_coded_responses = hard_coded_responses

    def get_name(self) -> str:
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Post Acceptance Action"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender responds to the user after user accepts a recommended item"

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], AcceptRecommendation):
                    return self.priority_score_range[0] + (goal["utterance_index"]-0.5) / len(state_manager.get("conv_history")) * (self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def get_prompt_response(self, state_manager: StateManager) -> str | None:
        """
        Return prompt based recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: prompt based recommender's response corresponding to this action
        """
        return None

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        
        for response_dict in self._hard_coded_responses:
            if response_dict['action'] == 'PostAcceptanceAction':
                return response_dict['response']

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.
        :return: whether hard coded response exists or not.
        """
        return True

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
