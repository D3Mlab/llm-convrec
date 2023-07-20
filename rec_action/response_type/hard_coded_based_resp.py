from state.state_manager import StateManager

class HardCodedBasedResponse:
    """
    Abstract class representing the hard coded based response
    """
    def get(self, state_manager: StateManager) -> str:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """
        raise NotImplementedError()