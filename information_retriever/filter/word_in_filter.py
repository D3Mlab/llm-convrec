from information_retriever.filter.filter import Filter
from state.state_manager import StateManager
import pandas as pd


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
