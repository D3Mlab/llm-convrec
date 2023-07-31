from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
import pandas as pd
import logging

logger = logging.getLogger('filter')


class WordInFilter(Filter):
    """
    Responsible to do filtering by checking whether singular / plural form of a word
    in the constraint is in the specified metadata field or singular / plural form of
    a word in the specified metadata field is in the constraint.

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

    def _does_item_match_constraint_fully(self, row_of_df: pd.Series, constraint_values: list[str]) -> bool:
        """
        Return true if for all constraint values, a word in the constraint matches partially with a word
        in the specified metadata field or a word in the specified metadata field
        matches partially with a value in the constraint, false otherwise.

        A constraint value is considered to be matching if...
         - constraint value is substring of at least one metadata value
         - at least one metadata value is substring of constraint value
         - plural form of constraint value is substring of at least one metadata value
         - at least one plural form of metadata value is substring of constraint value

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
                constraint_value_lower_stripped = constraint_value.lower().strip()
                metadata_field_value_lower_stripped = metadata_field_value.lower().strip()

                if constraint_value_lower_stripped in metadata_field_value_lower_stripped \
                        or metadata_field_value_lower_stripped in constraint_value_lower_stripped \
                        or self._convert_to_plural(
                            constraint_value_lower_stripped) in metadata_field_value_lower_stripped \
                        or self._convert_to_plural(
                            metadata_field_value_lower_stripped) in constraint_value_lower_stripped:
                    is_constraint_in_metadata = True
                    break
            if not is_constraint_in_metadata:
                return False

        return True

    def _does_item_match_constraint_partially(self, row_of_df: pd.Series, constraint_values: list[str]) -> bool:
        """
        Return true if there exists constraint value such that it matches partially with a word
        in the specified metadata field or a word in the specified metadata field
        matches partially with value in the constraint, false otherwise.

        A constraint value is considered to be matching if...
         - constraint value is substring of at least one metadata value
         - at least one metadata value is substring of constraint value
         - plural form of constraint value is substring of at least one metadata value
         - at least one plural form of metadata value is substring of constraint value

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
                constraint_value_lower_stripped = constraint_value.lower().strip()
                metadata_field_value_lower_stripped = metadata_field_value.lower().strip()

                if constraint_value_lower_stripped in metadata_field_value_lower_stripped\
                        or metadata_field_value_lower_stripped in constraint_value_lower_stripped\
                        or self._convert_to_plural(
                                constraint_value_lower_stripped) in metadata_field_value_lower_stripped\
                        or self._convert_to_plural(
                                metadata_field_value_lower_stripped) in constraint_value_lower_stripped:
                    return True

        return False

    @staticmethod
    def _convert_to_plural(word: str) -> str:
        """
        Try to convert the word to plural.

        :param word: word to be converted to plural
        :return: word in plural form
        """
        plural_rules = [
            (["s", "sh", "ch", "x", "z"], "es"),
            (["ay", "ey", "iy", "oy", "uy"], "s"),
            (["y"], "ies")
        ]

        for rule in plural_rules:
            endings, plural_ending = rule
            for end in endings:
                if word.endswith(end):
                    return word.rstrip(end) + plural_ending

        return word + "s"
