from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager


class ExactWordMatchingChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether a word in the constraint matches exactly with a word in the specified metadata field
    or a word in the specified metadata field matches exactly with a word in the constraint
    (case insensitive, ignore spaces).

    :param constraint_key: constraint key of interest
    :param metadata_field: metadata field of interest
    """

    _constraint_key: str
    _metadata_field: str

    def __init__(self, constraint_key: str, metadata_field: str) -> None:
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field

    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if the item match the constraint, false otherwise.
        If the constraint in interest is empty, it will return true.
        Might not work well if the value in the metadata filed is a dictionary.

        :param state_manager: current state
        :param item_metadata: item's metadata
        :return: true if the item match the constraint, false otherwise
        """
        constraint_values = state_manager.get('hard_constraints').get(self._constraint_key)
        item_metadata_field_values = item_metadata[self._metadata_field].split(",")

        if constraint_values is None:
            return True

        for metadata_field_value in item_metadata_field_values:
            for constraint_value in constraint_values:
                if constraint_value == metadata_field_value:
                    return True

        return False
