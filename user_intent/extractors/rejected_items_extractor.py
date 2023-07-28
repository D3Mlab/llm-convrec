from information_retriever.item.recommended_item import RecommendedItem
from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from jinja2 import Environment, FileSystemLoader, Template


class RejectedItemsExtractor:
    """
    Class responsible for extracting items rejected by the user.

    :param llm_wrapper: LLM used to extract items
    :param domain: domain of recommendation
    :param rejected_items_fewshots: few shots we use with prompts
    :param config: config of the system
    """

    _llm_wrapper: LLMWrapper
    _domain: str
    _fewshots: list
    template: Template

    def __init__(self, llm_wrapper: LLMWrapper, domain: str, rejected_items_fewshots: list, config: dict):
        self._llm_wrapper = llm_wrapper
        self._domain = domain
        self._fewshots = rejected_items_fewshots

        env = Environment(loader=FileSystemLoader(
            config['ITEMS_EXTRACTOR_PROMPT_PATH']))
        self.template = env.get_template(
            config['REJECTED_ITEMS_EXTRACTOR_PROMPT_FILENAME'])

    def extract(self, conv_history: list[Message], all_mentioned_items: list[RecommendedItem],
                recently_mentioned_items: list[RecommendedItem]) -> list[RecommendedItem]:
        """
        Extract rejected items from the most recent user's input.

        :param conv_history: past messages
        :param all_mentioned_items: all previously mentioned items
        :param recently_mentioned_items: most recently mentioned items
        :return: list of items rejected by the user
        """
        prompt = self._generate_prompt(
            conv_history, all_mentioned_items, recently_mentioned_items)
        llm_response = self._llm_wrapper.make_request(prompt)
        item_names = {item.strip().casefold()
                            for item in llm_response.split(',')}
        result = []

        for item in all_mentioned_items:
            if item.get_name().casefold() in item_names:
                result.append(item)

        return result

    def _generate_prompt(self, conv_history: list[Message], all_mentioned_items: list[RecommendedItem],
                         recently_mentioned_items: list[RecommendedItem]):
        """
        Generate and return prompt for extracting rejected items.

        :param conv_history: past messages in the conversation
        :return: prompt for extracting rejected items.
        """
        curr_user_input = conv_history[-1].get_content() if len(conv_history) >= 1 else ""

        return self.template.render(user_input=curr_user_input,
                                    recently_mentioned_items=[item.get_name() for item in
                                                                   recently_mentioned_items],
                                    all_mentioned_items=[item.get_name() for item in
                                                              all_mentioned_items],
                                    few_shots=self._fewshots,
                                    domain=self._domain)
