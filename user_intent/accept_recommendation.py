from state.state_manager import StateManager
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from user_intent.user_intent import UserIntent

from itertools import chain
from jinja2 import Environment, FileSystemLoader, Template


class AcceptRecommendation(UserIntent):
    """
    Class representing the Accept Recommendation user intent.

    :param accepted_items_extractor: object used to extract accepted items
    :param few_shots: few shot examples used in the prompt
    :param domain: domain of the recommendation (e.g. "restaurants")
    :param config: config of the system
    """

    _accepted_items_extractor: AcceptedItemsExtractor
    _few_shots: list[dict]
    _domain: str
    _template: Template

    def __init__(self, accepted_items_extractor: AcceptedItemsExtractor,
                 few_shots: list[dict], domain: str, config: dict):
        self._accepted_items_extractor = accepted_items_extractor

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self._template = env.get_template(
            config['ACCEPT_RECOMMENDATION_PROMPT_FILENAME'])
        
        self._few_shots = few_shots
        self._domain = domain

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

    def update_state(self, curr_state: StateManager):
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """

        if curr_state.get("recommended_items") is not None:
            all_mentioned_items = list(chain.from_iterable(
                curr_state.get('recommended_items')))
        else:
            all_mentioned_items = []

        items = self._accepted_items_extractor.extract(
            curr_state.get("conv_history"),
            all_mentioned_items,
            [] if curr_state.get(
                "curr_items") is None else curr_state.get("curr_items")
        )
        if curr_state.get('accepted_items') is None:
            curr_state.update('accepted_items', [])
        curr_state.get('accepted_items').extend(items)
        curr_state.get("updated_keys")['accepted_items'] = True

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of accept recommendation 

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self._template.render(user_input=user_input, few_shots=self._few_shots, domain=self._domain)
        return prompt

        
