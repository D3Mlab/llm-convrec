from state.state_manager import StateManager


class RecAction:
    """
    Abstract class representing the recommender action from WWW-21 paper.
    """
    priority_score_range: tuple[float, float]

    def __init__(self, priority_score_range: tuple[float, float]) -> None:
        self.priority_score_range = priority_score_range

    def get_name(self) -> str:
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        raise NotImplementedError()

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        raise NotImplementedError()

    def get_prompt(self, state_manager: StateManager) -> str | None:
        """
        Return prompt that can be inputted to LLM to produce recommender's response. 
        Return None if it doesn't exist. 

        :param state_manager: current state representing the conversation
        :return: prompt that can be inputted to LLM to produce recommender's response or None if it doesn't exist. 
        """
        raise NotImplementedError()

    def get_hard_coded_response(self, state_manager: StateManager) -> str | list | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        raise NotImplementedError()

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.
        :return: whether hard coded response exists or not.
        """
        raise NotImplementedError()

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        raise NotImplementedError()

    def update_state(self, state_manager: StateManager, response: str, **kwargs) -> None:
        """
        Updates the state based off of recommenders response

        :param state_manager: current state representing the conversation
        :param response: recommender response msg that is returned to the user
        :param **kwargs: misc. arguments 

        :return: none
        """

        raise NotImplementedError()