import numpy as np
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager


class Filter:
    """
    Responsible to return item ids that must be kept.

    :param metadata_wrapper: metadata wrapper
    """

    _metadata_wrapper: MetadataWrapper

    def __init__(self, metadata_wrapper: MetadataWrapper) -> None:
        self._metadata_wrapper = metadata_wrapper

    def filter(self, checkers: list[Checker], state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.

        :param checkers: check whether the item match all the conditions
        :param state_manager: current state
        :return: item ids that must be kept
        """
        num_item = self._metadata_wrapper.get_num_item()
        item_id_to_keep = []
        for index in range(num_item):
            item_metadata_dict = self._metadata_wrapper.get_item_dict_from_index(index)

            if self._should_keep_item(checkers, state_manager, item_metadata_dict):
                item_id_to_keep.append(item_metadata_dict['item_id'])

        return np.ndarray(item_id_to_keep)

    @staticmethod
    def _should_keep_item(checkers: list[Checker], state_manager: StateManager,
                          item_metadata_dict: dict) -> bool:
        """
        Return true if the item should be kept, false otherwise.

        :param checkers: check whether the item match all the conditions
        :param state_manager: current state
        :param item_metadata_dict: item metadata dictionary
        :return: true if the item should be kept, false otherwise
        """
        for checker in checkers:
            if not checker.check(state_manager, item_metadata_dict):
                return False
        return True
