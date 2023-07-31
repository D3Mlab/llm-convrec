from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
import pandas as pd
import logging

logger = logging.getLogger('filter')


class ExactWordMatchingFilter(Filter):
    """
    Responsible to do filtering by checking
    whether a word in the constraint matches exactly with a word in the specified metadata field
    or a word in the specified metadata field matches exactly with a word in the constraint
    (case insensitive).

    :param constraint_keys: constraint key of interest
    :param metadata_field: metadata field of interest
    """

    _constraint_keys: list[str]
    _metadata_field: str

    def __init__(self, constraint_keys: list[str], metadata_field: str) -> None:
        self._constraint_keys = constraint_keys
        self._metadata_field = metadata_field

    def filter(self, state_manager: StateManager,
               metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Return a filtered version of metadata pandas dataframe.

        :param state_manager: current state
        :param metadata: items' metadata
        :return: filtered version of metadata pandas dataframe
        """
        constraint_values = []
        for constraint_key in self._constraint_keys:
            constraint_value = state_manager.get('hard_constraints').get(constraint_key)

            if constraint_value is not None:
                constraint_values.extend(constraint_value)

        if not constraint_values:
            return metadata

        item_match = metadata.apply(
            self._does_item_match_constraint_fully, args=(constraint_values,), axis=1)

        filtered_metadata = metadata.loc[item_match]
        if filtered_metadata.shape[0] == 0:
            logger.debug("Partial filtering applied")
            item_match = metadata.apply(
                self._does_item_match_constraint_partially, args=(constraint_values,), axis=1)
            filtered_metadata = metadata.loc[item_match]

        return filtered_metadata

    def _does_item_match_constraint_fully(self,  row_of_df: pd.Series, constraint_values: list[str]) -> bool:
        """
        Return true if for all constraint values, a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :return: true if the item match the constraint, false otherwise
        """
        item_metadata_field_values = row_of_df[self._metadata_field]

        if not isinstance(item_metadata_field_values, list):
            if isinstance(item_metadata_field_values, str):
                item_metadata_field_values = item_metadata_field_values.split(",")
            else:
                return True

        for constraint_value in constraint_values:
            is_constraint_in_metadata = False
            for metadata_field_value in item_metadata_field_values:
                if constraint_value.lower().strip() == metadata_field_value.lower().strip():
                    is_constraint_in_metadata = True
                    break
            if not is_constraint_in_metadata:
                return False
        return True

    def _does_item_match_constraint_partially(self, row_of_df: pd.Series, constraint_values: list[str]) -> bool:
        """
        Return true if there exists constraint value such that a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :return: true if the item match the constraint, false otherwise
        """
        item_metadata_field_values = row_of_df[self._metadata_field]

        if not isinstance(item_metadata_field_values, list):
            if isinstance(item_metadata_field_values, str):
                item_metadata_field_values = item_metadata_field_values.split(",")
            else:
                return True

        for metadata_field_value in item_metadata_field_values:
            for constraint_value in constraint_values:

                if constraint_value.lower().strip() == metadata_field_value.lower().strip():
                    return True

        return False
