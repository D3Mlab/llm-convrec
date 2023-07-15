from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class LocationChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether the item is within max_distance from the location ( or one of the location)
    specified by the user
    """

    def check(self, state_manager: StateManager, item_metadata: dict):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError