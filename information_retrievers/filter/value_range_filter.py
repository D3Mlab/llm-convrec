from information_retrievers.filter.filter import Filter
from state.state_manager import StateManager
import pandas as pd
import re


class ValueRangeFilter(Filter):
    """
    Responsible to check whether the item match the constraint by checking
    whether one of the values in the specified metadata field is within one of the value range of the constraint.

    :param constraint_key: constraint key of interest
    :param metadata_field: metadata field of interest
    """

    _constraint_key: str
    _metadata_field: str

    def __init__(self, constraint_key: str, metadata_field: str) -> None:
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field

    def filter(self, state_manager: StateManager,
               filtered_metadata: pd.DataFrame) -> pd.DataFrame:

        constraint_values = state_manager.get('hard_constraints').get(self._constraint_key)

        if constraint_values is None:
            return filtered_metadata

        filtered_metadata['does_item_match_constraint'] = filtered_metadata.apply(
            self._does_item_match_constraint, args=tuple(constraint_values), axis=1)
        filtered_metadata = filtered_metadata.loc[filtered_metadata['does_item_match_constraint']]
        filtered_metadata.drop('does_item_match_constraint', axis=1)

        return filtered_metadata

    def _does_item_match_constraint(self, row_of_df: pd.Series, constraint_values: list[str]) -> bool:
        """
        Return true if a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :return: true if the item match the constraint, false otherwise
        """
        item_metadata_field_values = row_of_df[self._metadata_field]

        for value_range in constraint_values:
            value_range_list = self._remove_non_numeric(value_range).split("-")
            for metadata_field_value in item_metadata_field_values:
                if metadata_field_value >= value_range_list[0] and metadata_field_value >= value_range_list[1]:
                    return True

        return False

    @staticmethod
    def _remove_non_numeric(text: str):
        return re.sub(r'[^0-9-]', '', text)
