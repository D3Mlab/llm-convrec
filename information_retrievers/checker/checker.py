from state.state_manager import StateManager

class Checker:
    """
    Responsible to check whether the item match the constraint.
    """
    def check(self, state_manager: StateManager, item_metadata: dict):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError