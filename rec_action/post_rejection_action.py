from textwrap import dedent

from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.reject_recommendation import RejectRecommendation
from state.message import Message


class PostRejectionAction(RecAction):
    """
    Class representing Answer recommender action.
    """

    def __init__(self, priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)

    def get_name(self) -> str:
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "PostRejectanceAction"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender responds to the user after user rejects a recommended restaurant"

    def get_response_info(self, state_manager: StateManager) -> dict:
        """
        Returns recommender's response corresponding to this recommender action based on the given state.
        It asks LLM to generate recommendation based on the current state.

        :param state_manager: current state representing the conversation
        :return: recommender's response corresponding to this recommender action based on the current state.
        """
        return {"predefined_response": self.get_hard_coded_response(state_manager)}

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], RejectRecommendation):
                    return self.priority_score_range[0] + (goal["utterance_index"]-0.5) / len(state_manager.get("conv_history")) * (self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def get_prompt(self, state_manager: StateManager) -> str | None:
        """
        Return prompt that can be inputted to LLM to produce recommender's response. 
        Return None if it doesn't exist. 

        :param state_manager: current state representing the conversation
        :return: prompt that can be inputted to LLM to produce recommender's response or None if it doesn't exist. 
        """
        return None

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        return "I'm sorry that you did not like the recommendation. Is there anything else I can assist you with?"

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
