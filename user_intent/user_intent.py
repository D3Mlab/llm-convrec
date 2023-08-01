from state.state_manager import StateManager


class UserIntent:
    """
    Abstract class representing the user intent from WWW-21 paper.
    """

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        raise NotImplementedError()

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        raise NotImplementedError()

    def update_state(self, curr_state: StateManager):
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        raise NotImplementedError()
    
    def get_prompt_for_classification(self, curr_state:StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        
        raise NotImplementedError()



