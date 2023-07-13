from typing import Any

from state.message import Message


class StateManager:
    """
    Abstract class representing the state for the conversation, where each field (except for conversation history)
    is stored as key-value pair.
    """

    def get(self, key: str) -> Any:
        """
        Return the value of the field corresponding to the given key

        :param key: key corresponding to the field
        :return: value of the field corresponding to the given key
        """
        raise NotImplementedError()

    def update(self, key: str, value: Any) -> None:
        """
        Update the field for this state corresponding to the given key to the given value.

        :param key: key corresponding to the updated field
        :param value: new value for the updated field
        """
        raise NotImplementedError()

    def update_conv_history(self, message: Message) -> None:
        """
        Add most recent message to the conversation history.

        :param message: most recent message
        """
        raise NotImplementedError()

    def store_user_intents(self, user_intents: list) -> None:
        """
        Store the given user intents and their associated data to the state. 

        :param user_intents: user intents corresponding to the most recent user's input 
        """
        raise NotImplementedError()

    def store_rec_actions(self, rec_actions: list) -> None:
        """
        Store the given recommender actions and their associated data to the state. 

        :param rec_actions: recommender actions corresponding to the current user intents
        """
        raise NotImplementedError()

    def store_response(self, response: str, **kwargs) -> None:
        """
        Store the recommender response 

        :param response: recommender response corresponsing to the most recent user input
        :param **kwargs: additional parameters needed to update the state
        """
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """
        Convert this state to dictionary and return them. 

        :return: dictionary representation of the state
        """
        raise NotImplementedError()

    def reset_state(self) -> None:
        """
        Reset state.
        """
        raise NotImplementedError()
