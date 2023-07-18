from information_retrievers.filter.filter import Filter
from state.state_manager import StateManager
from information_retrievers.item.recommended_item import RecommendedItem
import pandas as pd


class ItemFilter(Filter):
    """
    Responsible to check the item is not in the item list in interest in state manager.

    :param key_in_state_manager: key of interest in state manager
    :param metadata_field: metadata field of interest
    """

    _key_in_state_manager: str
    _metadata_field: str

    def __init__(self, key_in_state_manager: str, metadata_field: str) -> None:
        self._key_in_state_manager = key_in_state_manager
        self._metadata_field = metadata_field

    def filter(self, state_manager: StateManager,
               filtered_metadata: pd.DataFrame) -> pd.DataFrame:
        item_list = state_manager.get(self._key_in_state_manager)

        if item_list is None:
            return filtered_metadata

        filtered_metadata['is_item_not_in_item_list'] = filtered_metadata.apply(
            self._is_item_not_in_item_list, args=tuple(item_list), axis=1)
        filtered_metadata = filtered_metadata.loc[filtered_metadata['is_item_not_in_item_list']]
        filtered_metadata.drop('is_item_not_in_item_list', axis=1)

        return filtered_metadata

    def _is_item_not_in_item_list(self, row_of_df: pd.Series, item_list: list[RecommendedItem]) -> bool:
        """
        Return true if a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :return: true if the item match the constraint, false otherwise
        """
        item_metadata_field_value = row_of_df[self._metadata_field]

        for item in item_list:
            if self._metadata_field == "item_id":
                if item.get_id() == item_metadata_field_value:
                    return False

            elif self._metadata_field == "name":
                if item.get_name() == item_metadata_field_value:
                    return False

        return True
