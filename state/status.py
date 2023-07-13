from state.state_manager import StateManager


class Status:
    """
    Abstract class representing the status for a constraint
    """
    _constraint: str
    _status_types: list[str]
    _state_key: str
    
    def __init__(self, constraint: str, key: str):
        self._constraint_name = constraint
        self._state_key = key
        
        self._status_types = ["invalid", "valid"]
        
        self.curr_status = "invalid"
            
    def update_status(self, curr_state: StateManager):
        """
        Update the status of the constraint

        :param curr_state: current representation of the state
        :return: None
        """
        raise NotImplementedError()

    def get_status(self):
        return self.curr_status