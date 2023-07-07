from state.state_manager import StateManager
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from user_intent.user_intent import UserIntent
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor

from itertools import chain
from jinja2 import Environment, FileSystemLoader
import yaml


class AcceptRecommendation(UserIntent):
    """
    Class representing the Accept Recommendation user intent.

    :param current_restaurants_extractor: object used to extract the restaurant that the user is referring to from the users input
    :param accepted_restaurants_extractor: object used to extract accepted restaurants

    """

    _current_restaurants_extractor: CurrentItemsExtractor
    _accepted_restaurants_extractor: AcceptedItemsExtractor

    def __init__(self, accepted_restaurants_extractor: AcceptedItemsExtractor, current_restaurants_extractor: CurrentItemsExtractor):
        self._accepted_restaurants_extractor = accepted_restaurants_extractor
        self._current_restaurants_extractor = current_restaurants_extractor

        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(
            config['ACCEPT_RECOMMENDATION_PROMPT_FILENAME'])

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        return "Accept Recommendation"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "User accepts recommended item"

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        # Update current restaurant
        reccommended_restaurants = curr_state.get("recommended_items")

        if reccommended_restaurants is not None and reccommended_restaurants != []:
            curr_res = self._current_restaurants_extractor.extract(
                reccommended_restaurants, curr_state.get("conv_history"))

            # If current restaurant is not an empty array then user talking about new restaurant
            if curr_res != []:
                curr_state.update("curr_items", curr_res)

        if curr_state.get("recommended_restaurants") is not None:
            all_mentioned_restaurants = list(chain.from_iterable(
                curr_state.get('recommended_restaurants')))
        else:
            all_mentioned_restaurants = []

        restaurants = self._accepted_restaurants_extractor.extract(
            curr_state.get("conv_history"),
            all_mentioned_restaurants,
            [] if curr_state.get(
                "curr_items") is None else curr_state.get("curr_items")
        )
        if curr_state.get('accepted_items') is None:
            curr_state.update('accepted_items', [])
        curr_state.get('accepted_items').extend(restaurants)
        curr_state.get("updated_keys")['accepted_items'] = True
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
