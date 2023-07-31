from information_retriever.metadata_wrapper import MetadataWrapper
from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
from information_retriever.item.recommended_item import RecommendedItem


class FilterApplier:
    """
    Responsible to return item ids that must be kept.

    :param metadata_wrapper: metadata wrapper
    :param filters: list of filters to apply
    """

    _metadata_wrapper: MetadataWrapper
    filters: list[Filter]

    def __init__(self, metadata_wrapper: MetadataWrapper, filters: list[Filter]) -> None:
        self._metadata_wrapper = metadata_wrapper
        self.filters = filters

    def apply_filter(self, state_manager: StateManager) -> list[int]:
        """
        Return a numpy array that has item ids that must be kept.

        :param state_manager: current state
        :return: item indices that must be kept
        """
        metadata = self._metadata_wrapper.get_metadata()

        for filter_obj in self.filters:
            if metadata.shape[0] == 0:
                break

            metadata = filter_obj.filter(state_manager, metadata)

        indices_list = metadata.index.tolist()
        return indices_list

    def filter_by_current_item(self, current_item: RecommendedItem) -> list[int]:
        """
        Return a numpy array that has item ids that must be kept.

        :param current_item: current item
        :return: item index that must be kept
        """
        metadata = self._metadata_wrapper.get_metadata()
        index = metadata.index[metadata['item_id'] == current_item.get_id()].tolist()
        return index
