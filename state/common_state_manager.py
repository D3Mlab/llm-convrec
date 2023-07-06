from typing import Any
from rec_action.answer import Answer
from rec_action.post_acceptance_action import PostAcceptanceAction
from rec_action.post_rejection_action import PostRejectionAction

from state.message import Message
from state.state_manager import StateManager
from user_intent.inquire import Inquire
from user_intent.accept_recommendation import AcceptRecommendation
from user_intent.reject_recommendation import RejectRecommendation

from user_intent.user_intent import UserIntent
from rec_action.rec_action import RecAction


class CommonStateManager(StateManager):
    """
    Implementation of StateManager that uses dictionary.

    :param data: default content of the state as a dictionary
    :param possible_goals: set of user intent that can be goals
    """

    _data: dict[str, Any]

    def __init__(self, possible_goals: set[UserIntent], default_goal: UserIntent = None, data=None):
        if data is None:
            data = {}

        self._data = data
        self._data['conv_history'] = []
        self._possible_goals = possible_goals
        self._default_goal = default_goal

    def get(self, key: str) -> Any:
        """
        Return the value of the field corresponding to the given key

        :param key: key corresponding to the field
        :return: value of the field corresponding to the given key
        """
        return self._data.get(key)

    def update(self, key: str, value: Any) -> None:
        """
        Update the field for this state corresponding to the given key to the given value.

        :param key: key corresponding to the updated field
        :param value: new value for the updated field
        """
        self._data[key] = value

    def update_conv_history(self, message: Message) -> None:
        """
        Add most recent message to the conversation history.

        :param message: most recent message
        """
        self._data['conv_history'].append(message)

    def store_user_intents(self, user_intents: list[UserIntent]) -> None:
        """
        Store the given user intents and their associated data to the state. 

        :param user_intents: user intents corresponding to the most recent user's input 
        """

        # move satisfied goals
        unsatisfied_goals = self.get("unsatisfied_goals")
        if unsatisfied_goals is not None:
            removed_unsatisfied_goals = []
            for goal in unsatisfied_goals:
                if self._is_goal_satisfied(goal):
                    if self.get("satisfied_goals") is None:
                        self._data["satisfied_goals"] = []
                    removed_unsatisfied_goals.append(goal)
                    self.get("satisfied_goals").append(goal)
            for goal in removed_unsatisfied_goals:
                unsatisfied_goals.remove(goal)

        self.update("updated_keys", {})

        num_goals = 0
        for user_intent in user_intents:
            if user_intent in self._possible_goals:
                num_goals += 1

        for user_intent in user_intents:
            if num_goals < 2 or (not isinstance(user_intent, AcceptRecommendation) and
                                 not isinstance(user_intent, RejectRecommendation)):
                self._update_goals(user_intent)
            user_intent.update_state(self)

    def store_rec_actions(self, rec_actions: list) -> None:
        """
        Store the given recommender actions and their associated data to the state. 

        :param rec_actions: recommender actions corresponding to the current user intents
        """
        self.update("current_rec_actions", rec_actions)

    def store_response(self, response: str, **kwargs) -> None:
        """
        Store the recommender response 

        :param response: recommender response corresponsing to the most recent user input
        :param **kwargs: additional parameters needed to update the state

        """

        rec_actions: list[RecAction] = self.get("current_rec_actions")

        if rec_actions is not None:
            for rec_action in rec_actions:
                rec_action.update_state(self, response, **kwargs)
        else:
            message = Message("recommender", response)
            self.update_conv_history(message)

    def to_dict(self) -> dict:
        """
        Convert this state to dictionary and return them. 

        :return: dictionary representation of the state
        """
        return self._data.copy()

    def _update_goals(self, user_intent: UserIntent) -> None:
        """
        Update goals in the state, by adding the given user intent to the state if
        the given user intent can be goals (i.e. in self._possible_goals).

        :param user_intent: new user intent
        :return: None
        """

        if self.get("unsatisfied_goals") is None:
            self.update("unsatisfied_goals", [{
                "user_intent": self._default_goal,
                "utterance_index": len(self.get("conv_history")) - 1
            }])

        if user_intent in self._possible_goals:
            self.get("unsatisfied_goals").append({
                "user_intent": user_intent,
                "utterance_index": len(self.get("conv_history")) - 1
            })

    def _is_goal_satisfied(self, goal: dict):
        """
        Returns whether the given goal is satisfied

        :param goal: goal that we are checking
        :return: whether the given goal is satisfied
        """
        if self.get("current_rec_actions") is not None:
            for rec_action in self.get("current_rec_actions"):
                if isinstance(goal["user_intent"], Inquire) and isinstance(rec_action, Answer) or isinstance(goal["user_intent"], AcceptRecommendation) and isinstance(rec_action, PostAcceptanceAction) or isinstance(goal["user_intent"], RejectRecommendation) and isinstance(rec_action, PostRejectionAction):
                    return True
        return False

    def __str__(self) -> str:
        return str(self._data)
