from state.state_manager import StateManager

from information_retriever.item.recommended_item import RecommendedItem
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.information_retrieval import InformationRetrieval
from rec_action.response_type.recommend_resp import RecommendResponse

from intelligence.llm_wrapper import LLMWrapper
import logging
from jinja2 import Environment, FileSystemLoader, Template
from warning_observer import WarningObserver
from utility.thread_utility import start_thread
import threading

from typing import Any

logger = logging.getLogger('recommend')


class RecommendPromptBasedResponse(RecommendResponse):
    """
    Class representing the prompt based response for recommend

    :param llm_wrapper: wrapper of LLM used to generate response
    :param filter_applier: object used to apply filter items
    :param information_retriever: object used to retrieve item reviews based on the query
    :param domain: domain of the recommendation (e.g. restaurants)
    :param hard_coded_responses: list that defines every hard coded response
    :param config: config for this system
    :param constraint_categories: list of dictionaries that defines the constraint details
    :param explanation_metadata_blacklist: list of metadata keys that should be ignored in recommendation explanation
    :param observers: observers that gets notified when reviews must be summarized, so it doesn't exceed
    """

    _llm_wrapper: LLMWrapper
    _filter_applier: FilterApplier
    _information_retriever: InformationRetrieval
    _hard_coded_responses: list[dict]
    _constraint_categories: list[dict]
    _explanation_metadata_blacklist: list[str]
    _observers: list[WarningObserver]
    _topk_items: int
    _topk_reviews: int
    _convert_state_to_query_prompt: Template
    _explain_recommendation_prompt: Template
    _format_recommendation_prompt: Template
    _summarize_review_prompt: Template
    _query: str
    _item_indices: list[int]
    _enable_threading: str
    _current_recommended_items: list[RecommendedItem]

    def __init__(self, llm_wrapper: LLMWrapper, filter_applier: FilterApplier,
                 information_retriever: InformationRetrieval, domain: str, hard_coded_responses: list[dict],
                 config: dict, constraint_categories: list[dict], explanation_metadata_blacklist: list[str] = None,
                 observers: list[WarningObserver] = None):
        super().__init__(domain)

        if explanation_metadata_blacklist is None:
            explanation_metadata_blacklist = []

        self._filter_applier = filter_applier
        self._information_retriever = information_retriever
        self._llm_wrapper = llm_wrapper
        self._observers = observers
        self._hard_coded_responses = hard_coded_responses
        self._constraint_categories = constraint_categories
        self._explanation_metadata_blacklist = explanation_metadata_blacklist

        self._topk_items = int(config["TOPK_ITEMS"])
        self._topk_reviews = int(config["TOPK_REVIEWS"])

        env = Environment(loader=FileSystemLoader(
            config['RECOMMEND_PROMPTS_PATH']))
        self._convert_state_to_query_prompt = env.get_template(
            config['CONVERT_STATE_TO_QUERY_PROMPT_FILENAME'])
        self._explain_recommendation_prompt = env.get_template(
            config['EXPLAIN_RECOMMENDATION_PROMPT_FILENAME'])
        self._format_recommendation_prompt = env.get_template(
            config['FORMAT_RECOMMENDATION_PROMPT_FILENAME'])
        self._summarize_review_prompt = env.get_template(
            config['SUMMARIZE_REVIEW_PROMPT_FILENAME'])

        self._query = ""
        self._item_indices = []

        self._enable_preference_elicitation = config["ENABLE_PREFERENCE_ELICITATION"]

        if self._enable_preference_elicitation:
            self._unacceptable_similarity_range = config['UNACCEPTABLE_SIMILARITY_SCORE_RANGE']
            self._max_number_similar_items = config["MAX_NUMBER_SIMILAR_ITEMS"]
        else:
            self._unacceptable_similarity_range = 0
            self._max_number_similar_items = 1

        self._enable_threading = config['ENABLE_MULTITHREADING']

    def get(self, state_manager: StateManager) -> str:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """

        if self._enable_threading:
            state_to_query_thread = threading.Thread(
                target=self._get_query, args=(state_manager,))

            embedding_matrix_thread = threading.Thread(
                target=self._get_item_indices, args=(state_manager,))

            start_thread(
                [state_to_query_thread, embedding_matrix_thread])
        else:
            self._get_query(state_manager)
            self._get_item_indices(state_manager)
        try:
            current_recommended_items = self._information_retriever.get_best_matching_items(self._query,
                                                                                            self._topk_items,
                                                                                            self._topk_reviews,
                                                                                            self._item_indices,
                                                                                            self._unacceptable_similarity_range,
                                                                                            self._max_number_similar_items)

        except Exception as e:
            logger.debug(f'There is an error: {e}')

            for response_dict in self._hard_coded_responses:
                if response_dict['action'] == 'NoRecommendation':
                    return response_dict['response']

        # If too many similar items
        if self._has_similar_items(current_recommended_items) and self._enable_preference_elicitation:
            resp = "We have a few recommendations that match what you're looking for. Do you have any other " \
                   "preferences that can help us narrow down the options? "
            # Make it false so you are not querying to user again that there are too many recommendation
            self._enable_preference_elicitation = False

        else:
            self._current_recommended_items = [group[0] for group in current_recommended_items if len(group) != 0]

            explanation = self._get_explanation_for_each_item(state_manager)

            prompt = self._get_prompt_to_format_recommendation(state_manager, explanation)
            resp = self._llm_wrapper.make_request(prompt)

        return self._clean_llm_response(resp)

    def _get_query(self, state_manager: StateManager) -> None:
        """
        Get query from the state

        :param state_manager: current state representing the conversation
        :return: None
        """
        self._query = self.convert_state_to_query(state_manager)
        logger.debug(f'Query for Information Retrieval: {self._query}')

    def _get_item_indices(self, state_manager: StateManager) -> None:
        """
        Get filtered embedding matrix

        :param state_manager: current state representing the conversation
        :return: None
        """
        self._item_indices = \
            self._filter_applier.apply_filter(state_manager)

    @staticmethod
    def _has_similar_items(current_recommended_items: list[list[RecommendedItem]]) -> bool:
        """
        See if the current recommended items are similar or if they are different enough to recommend

        :param current_recommended_items: current recommended items from IR
        :return: bool
        """
        for group in current_recommended_items:
            if len(group) > 1:
                return True

        return False

    @staticmethod
    def _clean_llm_response(resp: str) -> str:
        """" 
        Clean the response from the llm
        
        :param resp: response from LLM
        :return: cleaned str
        """

        if '"' in resp:
            # get rid of double quotes (llm sometimes outputs it)
            resp = resp.replace('"', "")

        return resp.removeprefix('Response to user:').removeprefix('response to user:').strip()

    def convert_state_to_query(self, state_manager: StateManager) -> str:
        """
        Convert this state to query and return it. 

        :param state_manager: current state_manager
        :return: query used for information retriever
        """
        hard_constraints = state_manager.get('hard_constraints')
        data = state_manager.to_dict()
        soft_constraints = data.get('soft_constraints')
        prompt = self._convert_state_to_query_prompt.render(
            hard_constraints=hard_constraints, soft_constraints=soft_constraints, domain=self._domain)
        query = self._llm_wrapper.make_request(prompt)

        return query

    def get_current_recommended_items(self) -> list[RecommendedItem]:
        """
        Get most recently recommended items
        """
        return self._current_recommended_items

    def _get_prompt_to_format_recommendation(self, state_manager: StateManager, explanation: dict[str, str]) -> str:
        """
        Get the prompt to get recommendation text with explanation.

        :param state_manager: current state manager
        :param explanation: explanation of why recommending each recommendation
        :return: prompt to get recommendation text with explanation
        """
        item_names = ' and '.join(
            [f'{rec_item.get_name()}' for rec_item in self._current_recommended_items])
        explanation_str = ', '.join(
            [f'{key}: {val}' for key, val in explanation.items()])

        hard_constraints = {}
        soft_constraints = {}

        if state_manager.get('hard_constraints'):
            hard_constraints = state_manager.get('hard_constraints').copy()

        if state_manager.get('soft_constraints'):
            soft_constraints = state_manager.get('soft_constraints').copy()

        filtered_hard_constraints, filtered_soft_constraints = \
            self.get_constraints_for_explanation(hard_constraints, soft_constraints)

        constraints_str = ", ".join(list(filtered_hard_constraints.keys())) + ", ".join(
            list(filtered_soft_constraints.keys()))

        return self._format_recommendation_prompt.render(
            item_names=item_names, explanation=explanation_str, domain=self._domain, constraints_str=constraints_str)

    def _get_explanation_for_each_item(self, state_manager: StateManager) -> dict[str, str]:
        """
        Returns the explanation on why recommending each item

        :param state_manager: current state representing the conversation
        :return: explanation for each item stored in dict where key is item name and value is explanation
        """
        explanation = {}
        for rec_item in self._current_recommended_items:
            item_name = rec_item.get_name()
            hard_constraints = state_manager.get('hard_constraints').copy()

            data = state_manager.to_dict()
            soft_constraints = {}
            for key, value in data.items():
                if key == "soft_constraints":
                    soft_constraints = value
            metadata = self._get_metadata_of_rec_item(rec_item)
            reviews = rec_item.get_most_relevant_review()
            try:

                filtered_hard_constraints, filtered_soft_constraints = \
                    self.get_constraints_for_explanation(hard_constraints, soft_constraints)

                prompt = self._get_prompt_to_explain_recommendation(item_name, metadata, reviews,
                                                                    filtered_hard_constraints,
                                                                    filtered_soft_constraints)
                explanation[item_name] = self._llm_wrapper.make_request(
                    prompt)
            except Exception as e:
                logger.debug(f'There is an error: {e}')
                # this is very slow

                self._notify_observers()

                logger.debug("Reviews are too long, summarizing...")

                constraints = hard_constraints.copy()
                if soft_constraints is not None:
                    constraints.update(soft_constraints)
                summarized_reviews = []
                for review in rec_item.get_most_relevant_review():
                    summarize_review_prompt = self._get_prompt_to_summarize_review(
                        constraints, review)
                    summarized_review = self._llm_wrapper.make_request(
                        summarize_review_prompt)
                    summarized_reviews.append(summarized_review)

                filtered_hard_constraints, filtered_soft_constraints = \
                    self.get_constraints_for_explanation(hard_constraints, soft_constraints)

                prompt = self._get_prompt_to_explain_recommendation(item_name, metadata, summarized_reviews,
                                                                    filtered_hard_constraints,
                                                                    filtered_soft_constraints)
                explanation[item_name] = self._llm_wrapper.make_request(
                    prompt)

        return explanation

    def _get_prompt_to_explain_recommendation(self, item_names: str, metadata: str, reviews: list[str],
                                              hard_constraints: dict, soft_constraints: dict) -> str:
        """
        Get the prompt to get explanation.

        :param item_names: item name
        :param metadata: metadata of the item
        :param reviews: reviews of the item
        :param hard_constraints: hard constraints in current statemanager
        :param soft_constraints: soft constraints in current statemanager
        :return: prompt to get explanation
        """
        return self._explain_recommendation_prompt.render(
            item_names=item_names, metadata=metadata, reviews=reviews,
            hard_constraints=hard_constraints, soft_constraints=soft_constraints, domain=self._domain)

    def _get_prompt_to_summarize_review(self, constraints: dict, review: str) -> str:
        """
        Get prompt to summarize a review.

        :param constraints: both hard and soft constraints combined in current statemanager
        :param review: a review to be summarized
        :return: prompt to summarize a review
        """
        constraints_str = ', '.join(
            [f'{key}: {val}' for key, val in constraints.items()])
        return self._summarize_review_prompt.render(constraints=constraints_str, review=review, domain=self._domain)

    def _get_metadata_of_rec_item(self, recommended_item: RecommendedItem) -> str:
        """
        Get metadata of an item used for recommend

        :param recommended_item: recommended item whose metadata to be returned
        :return: string representation of metadata
        """
        attributes = ', '.join(
            [f'{key}: {val}' for key, val in recommended_item.get_mandatory_data().items() if key not in
             self._explanation_metadata_blacklist] +
            [f'{key}: {val}' for key, val in recommended_item.get_optional_data().items() if key not in
             self._explanation_metadata_blacklist])
        return attributes

    def _notify_observers(self) -> None:
        """
        Notify observers that there are some difficulties.
        """
        for observer in self._observers:
            observer.notify_warning()

    def get_constraints_for_explanation(self, hard_constraints: dict[str, Any], soft_constraints: dict[str, Any]) -> \
            tuple[dict[str, Any], dict[str, Any]]:
        """
        Return hard and soft constraints that should be inputted to prompt for generating explanation for
        recommendation.

        :param hard_constraints: hard constraints in the current state
        :param soft_constraints: soft constraints in the current state
        :return: hard and soft constraints that should be inputted to prompt for generating
                 explanation for recommendation.
        """
        filtered_hard_constraints = {}
        for constraint in self._constraint_categories:
            if constraint['in_explanation'] and constraint['key'] in hard_constraints:
                filtered_hard_constraints[constraint['key']] = hard_constraints[constraint['key']]

        if soft_constraints is not None:
            filtered_soft_constraints = {}
            for constraint in self._constraint_categories:
                if constraint['in_explanation'] and constraint['key'] in soft_constraints:
                    filtered_soft_constraints[constraint['key']] = soft_constraints[constraint['key']]
        else:
            filtered_soft_constraints = None
        return filtered_hard_constraints, filtered_soft_constraints
