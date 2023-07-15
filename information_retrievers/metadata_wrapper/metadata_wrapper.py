import numpy as np
from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class MetadataWrapper:
    """
    Metadata wrapper that is responsible to do filter and get item metadata as dictionary.
    """
    def filter(self, checkers: list[Checker], state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.
        """
        raise NotImplementedError()

    def get_item_dict(self, item_id: str) -> dict[str, str]:
        """
        Return item metadata as a dictionary from item id.
        """
        raise NotImplementedError()

    def should_keep_item(self, checkers: list[Checker], state_manager: StateManager,
                         item_metadata_dict: dict) -> bool:
        """
        Return true if the item should be kept, false otherwise.
        """
        for checker in checkers:
            if not checker.check(state_manager, item_metadata_dict):
                return False
        return True
