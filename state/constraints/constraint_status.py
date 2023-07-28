from state.state_manager import StateManager


class ConstraintStatus:
    """
    Abstract class representing the status for a constraint
    """
    _constraint_name: str
    _curr_status: str | None
    
    def __init__(self, constraint: str):
        self._constraint_name = constraint
        self._curr_status = None
            
    def update_status(self, curr_state: StateManager) -> None:
        """
        Update the status of the constraint

        :param curr_state: current representation of the state
        """
        raise NotImplementedError()

    def get_response_from_status(self) -> str | None:
        """
        Gets recommender response based off of constraints status. 
        Returns None if there are no response corresponding to the current status.

        :return: recommender response
        """
        raise NotImplementedError()

    def get_status(self) -> str | None:
        """
        Gets constraints status

        :return: constraint status
        """
        return self._curr_status
    
    def get_constraint_name(self):
        """
        Gets constraint name

        :return: constraint name
        """
        return self._constraint_name
