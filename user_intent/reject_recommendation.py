from itertools import chain

from state.state_manager import StateManager
from user_intent.extractors.rejected_items_extractor import RejectedItemsExtractor
from user_intent.user_intent import UserIntent
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from jinja2 import Environment, FileSystemLoader
import yaml


class RejectRecommendation(UserIntent):
    """
    Class representing the Reject Recommendation user intent.

    :param rejected_items_extractor: object used to extract rejected restaurants
    :param current_items_extractor: object used to extract the restaurant that the user is referring to from the users input
    """

    _current_items_extractor: CurrentItemsExtractor
    _rejected_items_extractor: RejectedItemsExtractor

    def __init__(self, rejected_items_extractor: RejectedItemsExtractor, current_items_extractor: CurrentItemsExtractor,few_shots: list[dict], domain: str, config: dict):
        self._rejected_items_extractor = rejected_items_extractor
        self._current_items_extractor = current_items_extractor

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(
            config['REJECT_RECOMMENDATION_PROMPT_FILENAME'])
        self._few_shots = few_shots
        self._domain = domain

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        return "Reject Recommendation"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "User rejects recommended item"

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        # Update current items
        reccommended_items = curr_state.get("recommended_items")

        if reccommended_items is not None and reccommended_items != []:
            curr_items = self._current_items_extractor.extract(
                reccommended_items, curr_state.get("conv_history"))

            # If current items are [] then just keep it the same
            if curr_items != []:
                curr_state.update("curr_items", curr_items)

        # Update rejected items
        if curr_state.get("recommended_items") is not None:
            all_mentioned_items = list(
                chain.from_iterable(curr_state.get("recommended_items")))
        else:
            all_mentioned_items = []

        items = self._rejected_items_extractor.extract(
            curr_state.get("conv_history"),
            all_mentioned_items,
            [] if curr_state.get(
                "curr_items") is None else curr_state.get("curr_items")
        )

        if curr_state.get('rejected_items') is None:
            curr_state.update('rejected_items', [])
        curr_state.get('rejected_items').extend(items)
        curr_state.get("updated_keys")['rejected_items'] = True
        return curr_state

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of accept recommendation 

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """

        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self.template.render(user_input=user_input)

        return prompt
