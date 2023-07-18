import numpy as np
from information_retrievers.metadata_wrapper import MetadataWrapper
from information_retrievers.filter.filter import Filter
from state.state_manager import StateManager
from information_retrievers.item.recommended_item import RecommendedItem
from domain_specific_config_loader import DomainSpecificConfigLoader


class FilterApplier:
    """
    Responsible to return item ids that must be kept.

    :param metadata_wrapper: metadata wrapper
    """

    _metadata_wrapper: MetadataWrapper
    _filters: list[Filter]

    def __init__(self, metadata_wrapper: MetadataWrapper) -> None:
        self._metadata_wrapper = metadata_wrapper
        domain_specific_config_loader = DomainSpecificConfigLoader()
        self._filters = domain_specific_config_loader.load_filters()

    def apply_filter(self, state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.

        :param state_manager: current state
        :return: item ids that must be kept
        """
        metadata = self._metadata_wrapper.get_metadata()
        item_id_list = metadata['item_id'].tolist()
        print(len(item_id_list))

        for filter_obj in self._filters:
            metadata = filter_obj.filter(state_manager, metadata)

        item_id_list = metadata['item_id'].tolist()
        print(len(item_id_list))
        return np.array(item_id_list)

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
