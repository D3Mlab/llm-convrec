from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from information_retriever.item.recommended_item import RecommendedItem

from string import punctuation
from jinja2 import Environment, FileSystemLoader, Template


class CurrentItemsExtractor:
    """
    Class used to extract the current item the user is referring to from the most recent user's input.

    :param llm_wrapper: LLMWrapper used to extract items
    :param domain: domain of recommendation
    :param curr_items_fewshots: few shot examples for the prompt
    """
    _llm_wrapper: LLMWrapper
    _domain: str
    _few_shots: list
    template: Template

    def __init__(self, llm_wrapper: LLMWrapper, domain: str, curr_items_fewshots: list, config: dict) -> None:
        self._llm_wrapper = llm_wrapper
        self._domain = domain
        self._few_shots = curr_items_fewshots
   
        env = Environment(loader=FileSystemLoader(config['ITEMS_EXTRACTOR_PROMPT_PATH']),
                          trim_blocks=True, lstrip_blocks=True)
        self.template = env.get_template(config['CURRENT_ITEMS_EXTRACTOR_PROMPT_FILENAME'])

    def extract(self, recommended_items: list[list[RecommendedItem]], conv_history: list[Message]) \
            -> list[RecommendedItem] | None:
        """
        Extract the current item from the most recent user's input in the conv_history and
        return it.

        :param recommended_items: current recommended items as a list of lists where each sublist are recommendations
                                  made in one turn.
        :param conv_history: current conversation history
        :return current restaurant
        """
        prompt = self._generate_restaurants_update_prompt(
            conv_history, recommended_items)

        llm_response = self._llm_wrapper.make_request(prompt)
        llm_response = self._clean_string(llm_response)
        curr_mentioned_items = self._get_objects_from_llm_response(
            recommended_items, llm_response)

        return curr_mentioned_items

    def _generate_restaurants_update_prompt(self, conv_history: list[Message],
                                            recommended_restaurants: list[list[RecommendedItem]]) -> str:
        """
        Generate and return prompt for extracting current restaurants the user is referring to from the most
        recent user's input in the conversation history.

        :param recommended_restaurants: recommended restaurants as a dictionary
        :param conv_history: current conversation history
        :return prompt for extracting current restaurant from the most recent user's input in the conversation history.
        """
        recc_res_names = [current_mentioned_restaurant.get_name() for recommended_restaurants_per_uttr in
                          recommended_restaurants for current_mentioned_restaurant in recommended_restaurants_per_uttr]

        curr_ment_res_names_str = ", ".join(recc_res_names)

        current_user_input = conv_history[-1].get_content()

        return self.template.render(user_input=current_user_input,
                                    curr_ment_item_names_str=curr_ment_res_names_str,
                                    domain=self._domain, few_shots=self._few_shots)

    def _clean_string(self, llm_response) -> str:
        """
        Get rid of punctuation and leading and trailing spaces
        :param llm_response: response from LLM
        :return string
        """
        return llm_response.strip(punctuation + " ")

    def _get_objects_from_llm_response(self, recommended_items: list[list[RecommendedItem]], llm_response: str) \
            -> list[RecommendedItem]:
        """
        Verify that the response is one of the recommended items and return the object

        :param recommended_items: current recommended items as a list of lists
        :param llm_response: response from LLM
        :return recommended items corresponding to the llm response
        """
        items = []

        if llm_response == 'None':
            return items

        llm_resp_items = llm_response.split(',')

        for recommended_items_per_turn in recommended_items:
            for recommended_item in recommended_items_per_turn:
                if recommended_item.get_name() in llm_resp_items:
                    items.append(recommended_item)

        return items
