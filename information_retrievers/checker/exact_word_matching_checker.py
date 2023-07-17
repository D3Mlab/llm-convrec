from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager


class ExactWordMatchingChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
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

    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if a word in the constraint matches exactly with a word
        in the specified metadata field or a word in the specified metadata field
        matches exactly with a word in the constraint, false otherwise.
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

        item_metadata_field_values = item_metadata[self._metadata_field]

        if constraint_values is None:
            return True

        for metadata_field_value in item_metadata_field_values:
            for constraint_value in constraint_values:
                if constraint_value.lower().strip() == metadata_field_value.lower().strip():
                    return True

        return False
