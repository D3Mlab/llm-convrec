from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class WordInChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether singular / plural form of a word in the constraint is in the specified metadata field
    or singular / plural form of a word in the specified metadata field is in the constraint.
    """

    def check(self, state_manager: StateManager, item_metadata: dict):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError