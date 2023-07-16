from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class ValueRangeChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether one of the values in the specified metadata field is within one of the value range of the constraint.
    """

    def __init__(self, constraint_key: str, metadata_field: str):
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field


    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if the item match the constraint, false otherwise.
        """
        constraint_values = state_manager.get('hard_constraints').get(self._constraint_key)
        item_metadata_field_values = item_metadata[self._metadata_field].split(",")

        for value_range in constraint_values:
            value_range_list = value_range.replace("$", "").replace(" ", "").split(",")
            for metadata_field_value in item_metadata_field_values:
                if metadata_field_value >= value_range_list[0] and metadata_field_value >= value_range_list[1]:
                    return True

        return False