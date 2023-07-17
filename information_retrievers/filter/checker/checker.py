from state.state_manager import StateManager


class Checker:
    """
    Responsible to check whether the item match the condition.
    """
    def check(self, state_manager: StateManager, item_metadata: dict) -> bool:
        """
        Return true if the item match the constraint, false otherwise.

        :param state_manager: current state
        :param item_metadata: item's metadata
        :return: true if the item match the condition, false otherwise
        """
        raise NotImplementedError
