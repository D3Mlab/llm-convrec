from state.state_manager import StateManager


class ConstraintsUpdater:

    """
    Class responsible for updating the constraints in the state, based on the user's input.
    """

    def update_constraints(self, state_manager: StateManager) -> None:
        """
        Update the hard and soft constraints based on the most recent user's input.

        :param state_manager: current state
        """
        raise NotImplementedError()
