import yaml

from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retrievers.item.recommended_item import RecommendedItem
from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from jinja2 import Environment, FileSystemLoader


class RejectedItemsExtractor:
    """
    Class responsible for extracting restaurants rejected by the user.

    :param llm_wrapper: LLM used to extract restaurants
    :param domain: domain of recommendation
    """

    _llm_wrapper: LLMWrapper
    _domain: str

    def __init__(self, llm_wrapper: LLMWrapper, domain: str):
        self._llm_wrapper = llm_wrapper
        self._domain = domain
        domain_specific_config_loader = DomainSpecificConfigLoader()
        self._fewshots = domain_specific_config_loader.load_rejected_items_fewshots()
        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(
            config['ITEMS_EXTRACTOR_PROMPT_PATH']))
        self.template = env.get_template(
            config['REJECTED_ITEMS_EXTRACTOR_PROMPT_FILENAME'])

    def extract(self, conv_history: list[Message], all_mentioned_restaurants: list[RecommendedItem],
                recently_mentioned_restaurants: list[RecommendedItem]) -> list[RecommendedItem]:
        """
        Extract rejected restaurants from the most recent user's input.

        :param conv_history: past messages
        :param all_mentioned_restaurants: all previously mentioned restaurants
        :param recently_mentioned_restaurants: most recently mentioned restaurants
        :return: list of restaurants rejected by the user
        """
        prompt = self._generate_prompt(
            conv_history, all_mentioned_restaurants, recently_mentioned_restaurants)
        llm_response = self._llm_wrapper.make_request(prompt)
        restaurant_names = {restaurant.strip().casefold()
                            for restaurant in llm_response.split(',')}
        result = []
        for restaurant in all_mentioned_restaurants:
            if restaurant.get_name().casefold() in restaurant_names:
                result.append(restaurant)
        return result

    def _generate_prompt(self, conv_history: list[Message], all_mentioned_restaurants: list[RecommendedItem],
                         recently_mentioned_restaurants: list[RecommendedItem]):
        """
        Generate and return prompt for extracting rejected restaurants.

        :param conv_history: past messages in the conversation
        :return: prompt for extracting rejected restaurants.
        """
        curr_user_input = conv_history[-1].get_content() if len(conv_history) >= 1 else ""

        return self.template.render(user_input=curr_user_input,
                                    recently_mentioned_items=[restaurant.get_name() for restaurant in
                                                                   recently_mentioned_restaurants],
                                    all_mentioned_items=[restaurant.get_name() for restaurant in
                                                              all_mentioned_restaurants],
                                    few_shots=self._fewshots,
                                    domain=self._domain)
