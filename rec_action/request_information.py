from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from state.constraints.constraint_status import ConstraintStatus
from user_intent.ask_for_recommendation import AskForRecommendation
from rec_action.response_type.request_information_hard_coded_resp import RequestInformationHardCodedBasedResponse
from state.message import Message


class RequestInformation(RecAction):
    """
    Class representing Request Information recommender action.

    :param constraint_statuses: objects that keep tracks the status of the constraints
    :param hard_coded_responses: object used to generate the response corresponding to this rec action
    :param request_info_resp: object used to generate the response corresponding to this rec action
    :param priority_score_range: range of scores for classifying recaction
    """
    _mandatory_constraints: list[list[str]]
    _constraint_statuses: list[ConstraintStatus]
    _request_info_resp: RequestInformationHardCodedBasedResponse

    def __init__(self, constraint_statuses: list[ConstraintStatus], hard_coded_responses: list[dict],
                 request_info_resp: RequestInformationHardCodedBasedResponse,
                 priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)
        self._constraint_statuses = constraint_statuses
        self._request_info_resp = request_info_resp
        self._mandatory_constraints = [response_dict['constraints'] for response_dict in hard_coded_responses
                                       if response_dict['action'] == 'RequestInformation'
                                       and response_dict['constraints'] != []]

    def get_name(self) -> str:
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Request Information"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender requests the user’s preference"

    def get_response(self, state_manager: StateManager) -> str | None:
        """
        Return recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: recommender's response corresponding to this action
        """
        return self._request_info_resp.get(state_manager)

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.

        :return: whether hard coded response exists or not.
        """
        return True

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
                if constraint.get_response_from_status() is not None:
                    is_ready = False
                    break

        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], AskForRecommendation) and not is_ready:
                    return self.priority_score_range[0] + goal["utterance_index"] / len(
                        state_manager.get("conv_history")) * (
                                       self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def update_state(self, state_manager: StateManager, response: str, **kwargs) -> None:
        """
        Updates the state based off of recommenders response

        :param state_manager: current state representing the conversation
        :param response: recommender response msg that is returned to the user
        :param kwargs: misc. arguments
        """
        message = Message("recommender", response)
        state_manager.update_conv_history(message)
