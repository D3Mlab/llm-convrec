from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
import pandas as pd
import re


class ValueRangeFilter(Filter):
    """
    Responsible to do filtering by checking whether one of the values
    in the specified metadata field is within one of the value range of the constraint.

    :param constraint_key: constraint key of interest
    :param metadata_field: metadata field of interest
    """

    _constraint_key: str
    _metadata_field: str

    def __init__(self, constraint_key: str, metadata_field: str) -> None:
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field

    def filter(self, state_manager: StateManager,
               metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Return a filtered version of metadata pandas dataframe.

        :param state_manager: current state
        :param metadata: items' metadata
        :return: filtered version of metadata pandas dataframe
        """
        constraint_values = state_manager.get('hard_constraints').get(self._constraint_key)

        if constraint_values is None:
            return metadata

        metadata['does_item_match_constraint'] = metadata.apply(
            self._does_item_match_constraint, args=(constraint_values,), axis=1)
        filtered_metadata = metadata.loc[metadata['does_item_match_constraint']]
        filtered_metadata = filtered_metadata.drop('does_item_match_constraint', axis=1)

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

        if not isinstance(item_metadata_field_values, list):
            if isinstance(item_metadata_field_values, str):

                if "-" in item_metadata_field_values:
                    item_metadata_field_values = item_metadata_field_values.split("-")

                    if len(item_metadata_field_values) != 2:
                        return True
                    else:
                        return self._do_value_ranges_overlap(constraint_values, item_metadata_field_values)

                else:
                    item_metadata_field_values = item_metadata_field_values.split(",")

            else:
                return True

        for value_range in constraint_values:
            value_range_list = re.sub(r'[^0-9-.]', '', value_range).split("-")

            if len(value_range_list) != 2:
                return True

            for metadata_field_value in item_metadata_field_values:
                metadata_field_value = re.sub(r'[^0-9.]', '', metadata_field_value)

                if float(value_range_list[0]) <= float(metadata_field_value) <= float(value_range_list[1]):
                    return True

        return False

    @staticmethod
    def _do_value_ranges_overlap(constraint_values: list[str], item_metadata_field_values: list[str]) -> bool:
        """
        Check whether one of the value range in constraint overlaps with the value range in metadata.

        :param constraint_values: value ranges in constraint, where each element is a value range
        that contains "-"
        :param item_metadata_field_values: value range in metadata field, where first element is the lower bound
        and second element is the upper bound
        :return: true if one of the value range in constraint overlaps with the value range in metadata,
        otherwise false
        """
        for value_range in constraint_values:
            value_range_list = re.sub(r'[^0-9-.]', '', value_range).split("-")

            if len(value_range_list) != 2:
                return True

            metadata_value_range_lower = re.sub(r'[^0-9.]', '', item_metadata_field_values[0])
            metadata_value_range_upper = re.sub(r'[^0-9.]', '', item_metadata_field_values[1])

            if (float(metadata_value_range_lower) <= float(value_range_list[1])
                and float(metadata_value_range_upper) >= float(value_range_list[0])) \
                    or (float(value_range_list[0]) <= float(metadata_value_range_upper)
                        and float(value_range_list[1]) >= float(metadata_value_range_lower)):
                return True

        return False
