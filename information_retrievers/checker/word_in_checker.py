from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager


class WordInChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether singular / plural form of a word in the constraint is in the specified metadata field
    or singular / plural form of a word in the specified metadata field is in the constraint.

    :param constraint_keys: constraint key of interest
    :param metadata_field: metadata field of interest
    """

    _constraint_keys: str
    _metadata_field: str

    def __init__(self, constraint_keys: str, metadata_field: str) -> None:
        self._constraint_keys = constraint_keys
        self._metadata_field = metadata_field

    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if singular / plural form of a word in the constraint is
        in the specified metadata field or singular / plural form of a word
        in the specified metadata field is in the constraint, false otherwise.
        If the constraint of interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :param state_manager: current state
        :param item_metadata: item's metadata
        :return: true if the item match the constraint, false otherwise
        """
        constraint_values = []
        for constraint_key in self._constraint_keys:
            constraint_value = state_manager.get('hard_constraints').get(constraint_key)
            if constraint_value is not None:
                constraint_values.append(constraint_value)

        item_metadata_field_values = item_metadata[self._metadata_field].split(",")

        if constraint_values is None:
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
