import re

from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.ask_for_recommendation import AskForRecommendation
from state.message import Message


class RequestInformation(RecAction):
    """
    Class representing Request Information recommender action.

    :param mandatory_constraints: set of constraints that are mandatory to recommend
    """
    _mandatory_constraints: list[list[str]]

    def __init__(self, priority_score_range: tuple[float, float] = (1, 10), mandatory_constraints: str = None,
                 specific_location_required: bool = True) -> None:
        super().__init__(priority_score_range)
        if mandatory_constraints is None:
            mandatory_constraints = [['location']]
        self._mandatory_constraints = mandatory_constraints
        self._specific_location_required = specific_location_required

    def get_name(self):
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Request Information"

    def get_description(self):
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender requests the user’s preference"

    def get_response_info(self, state_manager: StateManager):
        """
        Returns recommender's response corresponding to this recommender action based on the given state.
        It asks for location if location isn't found in the hard constraints of the state and asks for general
        preference otherwise.

        :param state_manager: current statemanager representing the conversation
        :return: recommender's response corresponding to this recommender action based on the current state.
        """
        return {"predefined_response": self.get_hard_coded_response(state_manager)}

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
        hard_constraints = state_manager.get("hard_constraints")

        for constraints in self._mandatory_constraints:
            formatted_constraints = [key for key in constraints]
            if len(formatted_constraints) > 1:
                formatted_constraints[-1] = f'or {formatted_constraints[-1]}'
            formatted_constraints = ', '.join(formatted_constraints).replace(', or ', ' or ')

            if hard_constraints is None or all(hard_constraints.get(constraint) is None or
                                               hard_constraints.get(constraint) == [] for constraint in constraints):
                return f"Can you provide the {formatted_constraints}?"
            elif 'location' in constraints and state_manager.get('location_type') == 'invalid':
                return "I am sorry, I don't know the given location. Can you provide different location?"
            elif 'location' in constraints and not state_manager.get("specific_location_asked") and \
                    self._specific_location_required and \
                    not state_manager.get('location_type') == 'specific':
                state_manager.update("specific_location_asked", True)
                return f'Could you provide a more specific location, such as the name of the street, avenue, or intersection?'

        return "Are there any additional preferences, requirements, or specific features you would like the restaurant to have?"

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

        if state_manager.get('location_type') is None or state_manager.get('location_type') == 'invalid' or \
                state_manager.get('location_type') == 'valid' and self._specific_location_required and \
                not state_manager.get("specific_location_asked"):
            is_ready = False

        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], AskForRecommendation) and not is_ready:
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
        message = Message("recommender", response)
        state_manager.update_conv_history(message)