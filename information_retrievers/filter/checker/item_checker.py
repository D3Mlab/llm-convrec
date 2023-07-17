from information_retrievers.filter.checker.checker import Checker
from state.state_manager import StateManager


class ItemChecker(Checker):
    """
    Responsible to check the item is not in the item list in interest in state manager.

    :param key_in_state_manager: key of interest in state manager
    :param metadata_field: metadata field of interest
    """

    _key_in_state_manager: str
    _metadata_field: str

    def __init__(self, key_in_state_manager: str, metadata_field: str) -> None:
        self._key_in_state_manager = key_in_state_manager
        self._metadata_field = metadata_field

    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if the item is not in the item list of interest, false otherwise.
        If the item list of interest is empty, it will return true.

        :param state_manager: current state
        :param item_metadata: item's metadata
        :return: true if the item is not in the item list in interest, false otherwise
        """
        item_list = state_manager.get(self._key_in_state_manager)
        item_metadata_field_value = item_metadata[self._metadata_field]

        if item_list is None:
            return True

        for item in item_list:
            if self._metadata_field == "item_id":
                if item.get_id() == item_metadata_field_value:
                    return False

            elif self._metadata_field == "name":
                if item.get_name() == item_metadata_field_value:
                    return False

        return True
