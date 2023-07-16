import numpy as np
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager


class Filter:

    def __init__(self, metadata_wrapper: MetadataWrapper):
        self._metadata_wrapper = metadata_wrapper

    def filter(self, checkers: list[Checker], state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.
        """
        num_item = self._metadata_wrapper.get_num_item()
        item_id_to_keep = []
        for index in range(num_item):
            item_metadata_dict = self._metadata_wrapper.get_item_dict_from_index(index)

            if self._should_keep_item(checkers, state_manager, item_metadata_dict):
                item_id_to_keep.append(item_metadata_dict['item_id'])

        return np.ndarray(item_id_to_keep)

    def _should_keep_item(self, checkers: list[Checker], state_manager: StateManager,
                          item_metadata_dict: dict) -> bool:
        """
        Return true if the item should be kept, false otherwise.
        """
        for checker in checkers:
            if not checker.check(state_manager, item_metadata_dict):
                return False
        return True
