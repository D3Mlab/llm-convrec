from rec_action.response_type.hard_coded_based_resp import HardCodedBasedResponse
from state.state_manager import StateManager

class AcceptHardCodedBasedResponse(HardCodedBasedResponse):
    """
    Class representing the hard code based response for post acceptance action
    """
    _hard_coded_responses: list[dict]
            
    def __init__(self, hard_coded_responses: list[dict]):
        self._hard_coded_responses = hard_coded_responses
    
    def get(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        
        for response_dict in self._hard_coded_responses:
            if response_dict['action'] == 'PostAcceptanceAction':
                return response_dict['response']
