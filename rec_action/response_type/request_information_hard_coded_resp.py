from rec_action.response_type.hard_coded_based_resp import HardCodedBasedResponse
from state.state_manager import StateManager


class RequestInformationHardCodedBasedResponse(HardCodedBasedResponse):
    """
    Class representing the hard code based response for request information
    """
    _hard_coded_responses: list[dict]

            
    def __init__(self, hard_coded_responses: list[dict]):
        self._hard_coded_responses = hard_coded_responses
    
    def get_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """ 
        
        hard_constraints = state_manager.get("hard_constraints")
        default_response = None

        for response_dict in self._hard_coded_responses:
            if response_dict['action'] == 'RequestInformation':
                constraints = response_dict['constraints']

                if not constraints:
                    default_response = response_dict['response']
                else:
                    if hard_constraints is None or all(hard_constraints.get(constraint) is None or
                                                       hard_constraints.get(constraint) == [] for constraint in constraints):
                        return response_dict['response']

        return default_response

