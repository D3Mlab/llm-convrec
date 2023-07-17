import numpy as np
from information_retrievers.metadata_wrapper import MetadataWrapper
from information_retrievers.filter.checker.checker import Checker
from state.state_manager import StateManager
from information_retrievers.item.recommended_item import RecommendedItem


class Filter:
    """
    Responsible to return item ids that must be kept.

    :param metadata_wrapper: metadata wrapper
    :param checkers: check whether the item match constraints
    """

    _metadata_wrapper: MetadataWrapper
    _checkers: list[Checker]

    def __init__(self, metadata_wrapper: MetadataWrapper, checkers: list[Checker]) -> None:
        self._metadata_wrapper = metadata_wrapper
        self._checkers = checkers

    def filter_by_checkers(self, state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.

        :param state_manager: current state
        :return: item ids that must be kept
        """
        num_item = self._metadata_wrapper.get_num_item()
        item_id_to_keep = []
        for index in range(num_item):
            item_metadata_dict = self._metadata_wrapper.get_item_dict_from_index(index)

            if self._should_keep_item(state_manager, item_metadata_dict):
                item_id_to_keep.append(item_metadata_dict['item_id'])

        return np.array(item_id_to_keep)

    @staticmethod
    def filter_by_current_item(current_items: list[RecommendedItem]) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.

        :param current_items: current items
        :return: item ids that must be kept
        """
        item_id_to_keep = []
        for current_item in current_items:
            item_id_to_keep.append(current_item.get_id())

        return np.array(item_id_to_keep)

    def _should_keep_item(self, state_manager: StateManager,
                          item_metadata_dict: dict) -> bool:
        """
        Return true if the item should be kept, false otherwise.

        :param state_manager: current state
        :param item_metadata_dict: item metadata dictionary
        :return: true if the item should be kept, false otherwise
        """
        for checker in self._checkers:
            if not checker.check(state_manager, item_metadata_dict):
                return False
        return True
