from state.state_manager import StateManager


class Status:
    """
    Abstract class representing the status for a constraint
    """
    _constraint_name: str
    _curr_status: str | None
    
    def __init__(self, constraint: str):
        self._constraint_name = constraint
        self._curr_status = None
            
    def update_status(self, curr_state: StateManager):
        """
        Update the status of the constraint

        :param curr_state: current representation of the state
        :return: None
        """
        raise NotImplementedError()

    def get_response_from_status(self):
        raise NotImplementedError()

    def get_status(self):
        return self._curr_status
    
    def get_constraint_name(self):
        return self._constraint_name