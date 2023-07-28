from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
from information_retriever.item.recommended_item import RecommendedItem
import pandas as pd
from itertools import chain


class ItemFilter(Filter):
    """
    Responsible to do filtering by checking the item is not in the item list in interest in state manager.

    :param key_in_state_manager: key of interest in state manager
    :param metadata_field: metadata field of interest
    """

    _key_in_state_manager: str
    _metadata_field: str

    def __init__(self, key_in_state_manager: str, metadata_field: str) -> None:
        self._key_in_state_manager = key_in_state_manager
        self._metadata_field = metadata_field

    def filter(self, state_manager: StateManager,
               metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Return a filtered version of metadata pandas dataframe.

        :param state_manager: current state
        :param metadata: items' metadata
        :return: filtered version of metadata pandas dataframe
        """
        item_nested_list = state_manager.get(self._key_in_state_manager)

        if item_nested_list is None or item_nested_list == []:
            return metadata

        metadata['is_item_not_in_item_list'] = metadata.apply(
            self._is_item_not_in_item_list, args=(item_nested_list,), axis=1)
        filtered_metadata = metadata.loc[metadata['is_item_not_in_item_list']]
        filtered_metadata = filtered_metadata.drop('is_item_not_in_item_list', axis=1)

        return filtered_metadata

    def _is_item_not_in_item_list(self, row_of_df: pd.Series, item_nested_list: list[list[RecommendedItem]]) -> bool:
        """
        Return true if a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :return: true if the item match the constraint, false otherwise
        """
        item_list = list(chain.from_iterable(item_nested_list))
        item_metadata_field_value = row_of_df[self._metadata_field]

        for item in item_list:
            if self._metadata_field == "item_id":
                if item.get_id().strip() == item_metadata_field_value.strip():
                    return False

            elif self._metadata_field == "name":
                if item.get_name().lower().strip() == item_metadata_field_value.lower().strip():
                    return False

        return True
