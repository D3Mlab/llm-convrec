from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.ask_for_recommendation import AskForRecommendation
from state.message import Message
import logging
from state.status import Status
from rec_action.response_type.hard_coded_based_resp import HardCodedBasedResponse
from rec_action.response_type.prompt_based_resp import PromptBasedResponse

logger = logging.getLogger('recommend')


class Recommend(RecAction):
    """
    Class representing Recommend recommender action.

    :param llm_wrapper: object to make request to LLM
    :param mandatory_constraints: constraints that state must have in order to recommend
    :param information_retriever: information retriever that is used to fetch restaurant recommendations
    :param priority_score_range: range of scores for classifying recaction
    """

    _mandatory_constraints: list[list[str]]
    _constraint_statuses: list[Status]
    _hard_coded_based_resp: HardCodedBasedResponse
    _prompt_based_resp: PromptBasedResponse

    def __init__(self,  constraint_statuses: list,
                 mandatory_constraints: list[list[str]], hard_coded_based_resp: HardCodedBasedResponse, prompt_based_resp: PromptBasedResponse,
                 priority_score_range=(1, 10)):
        super().__init__(priority_score_range)

        self._mandatory_constraints = mandatory_constraints
        self._constraint_statuses = constraint_statuses
        self._hard_coded_based_resp = hard_coded_based_resp
        self._prompt_based_resp = prompt_based_resp

    def get_name(self):
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Recommend"

    def get_description(self):
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "The recommender provides a recommendation either by directly presenting the item or asking the " \
               "user if she knows of the place."

    def get_prompt_response(self, state_manager: StateManager) -> str | None:
        """
        Return prompt based recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: prompt recommender's response corresponding to this action
        """
        return self._prompt_based_resp.get_response(state_manager)

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        return self._hard_coded_based_resp.get_response(state_manager)

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
        hard_constraints = state_manager.get("hard_constraints")
        is_ready = hard_constraints is not None and all(any(hard_constraints.get(key) is not None and
                                                            hard_constraints.get(key) != [] for key in lst)
                                                        for lst in self._mandatory_constraints)

        if self._constraint_statuses is not None:
            for constraint in self._constraint_statuses:
                if constraint.get_status() is None or constraint.get_status() == 'invalid':
                    is_ready = False
                    break
        
        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], AskForRecommendation) and is_ready:
                    return self.priority_score_range[0] + goal["utterance_index"] / len(
                        state_manager.get("conv_history")) * (
                        self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def update_state(self, state_manager: StateManager, response: str, **kwargs):
        """
        Updates the state based off of recommenders response

        :param state_manager: current state representing the conversation
        :param response: recommender response msg that is returned to the user
        :param **kwargs: misc. arguments

        :return: none
        """
        
        if (self.is_response_hard_coded()):
            current_recommended_items = self._hard_coded_based_resp.get_current_recommended_items()
        else:
            current_recommended_items= self._prompt_based_resp.get_current_recommended_items()

        message = Message("recommender", response)
        state_manager.update_conv_history(message)

        # Store recommended restaurants
        recommended_restaurants = state_manager.get('recommended_items')

        if recommended_restaurants is None:
            recommended_restaurants = []

        if current_recommended_items != []:
            recommended_restaurants.append(
                current_recommended_items)

            state_manager.update("recommended_items",
                                 recommended_restaurants)

            # Store new currrent restaurants
            state_manager.update("curr_items",
                                 current_recommended_items)
