import numpy as np
from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class MetadataWrapper:
    """
    Metadata wrapper that is responsible to get item metadata as dictionary.
    """

    def get_item_dict_from_id(self, item_id: str) -> dict[str, str]:
        """
        Return item metadata as a dictionary from item id.
        """
        raise NotImplementedError()

    def get_item_dict_from_index(self, index: int) -> dict[str, str]:
        """
        Return item metadata as a dictionary from index.
        """
        raise NotImplementedError()

    def get_num_item(self) -> int:
        """
        Return the number of items.
        """
        raise NotImplementedError()

