from state.state_manager import StateManager
from typing import Dict, Any

from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.information_retrieval import InformationRetrieval
from rec_action.response_type.prompt_based_resp import PromptBasedResponse
from rec_action.response_type.recommend_resp import RecommendResponse

from intelligence.llm_wrapper import LLMWrapper
import logging
from jinja2 import Environment, FileSystemLoader
from warning_observer import WarningObserver
from utility.thread_utility import start_thread
import threading

logger = logging.getLogger('recommend')

class RecommendPromptBasedResponse(RecommendResponse, PromptBasedResponse):
    """
    Class representing the prompt based response for recommend
    """
    _observers: list[WarningObserver]
    _llm_wrapper: LLMWrapper
    _filter_items: FilterApplier
    _information_retriever: InformationRetrieval
    _topk_items: str
    _topk_reviews: str
    _convert_state_to_query_prompt: str
    _explain_recommendation_prompt: str
    _format_recommendation_prompt: str
    _summarize_review_prompt: str

    def __init__(self, llm_wrapper: LLMWrapper, filter_items: FilterApplier,
                 information_retriever: InformationRetrieval, domain: str, hard_coded_responses: list[dict], config: dict, observers = None):

        super().__init__(domain)
        
        self._filter_items = filter_items
        self._information_retriever = information_retriever
        self._llm_wrapper = llm_wrapper
        self._observers = observers
        self._hard_coded_responses = hard_coded_responses

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
        
        self.query = ""
        self.explanation = {}
        self.item_ids = []
        
        self.enable_threading = config['ENABLE_MULTITHREADING']
       
    def get(self, state_manager: StateManager) -> str:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """
        
        if (self.enable_threading):
            state_to_query_thread = threading.Thread(
                target=self._get_query, args=(state_manager,))

            embedding_matrix_thread = threading.Thread(
                target=self._get_item_ids, args=(state_manager,))
            
            start_thread(
                [state_to_query_thread, embedding_matrix_thread])
        else:
            self._get_query(state_manager)
            self._get_item_ids(state_manager)

        try:
            if (self.enable_threading):
                explanation_thread = threading.Thread(
                    target=self._get_explanation, args=(state_manager,))

                reccommendation_thread = threading.Thread(
                    target=self._get_recommendation)

                start_thread(
                    [explanation_thread, reccommendation_thread])
            else:
                self._get_explanation(state_manager)
                self._get_recommendation()
                
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            
            for response_dict in self._hard_coded_responses:
                if response_dict['action'] == 'NoRecommendation':
                    return response_dict['response']
       
        prompt = self._get_prompt_to_format_recommendation()
        resp = self._llm_wrapper.make_request(prompt)
        
        return self._clean_llm_response(resp)

    def _get_query(self, state_manager: StateManager) -> None:
        """
        Get query from the state

        :param state_manager: current state representing the conversation
        :param ir_params: dict representing the parameters needed for IR
        :return: None
        """
        self.query = self.convert_state_to_query(state_manager)
        logger.debug(f'Query: { self.query}')
    
    def _get_item_ids(self, state_manager: StateManager) -> None:
        """
        Get filtered embedding matrix

        :param state_manager: current state representing the conversation
        :param ir_params: dict representing the parameters needed for IR
        :return: None
        """
        self.item_ids = \
            self._filter_items.apply_filter(state_manager)
    
    def _get_explanation(self, state_manager: StateManager) -> None:
        """
        Get explanation

        :param state_manager: current state representing the conversation
        :param explanation: set representing the explanation
        :return: None
        """
        self._get_explanation_for_each_item(state_manager)

    def _get_recommendation(self) -> None:
        """
        Get recommendation

        :param ir_params: dict representing the parameters needed for IR
        :return: None
        """
        self._current_recommended_items = self._information_retriever.get_best_matching_items(self.query, self._topk_items,
                                                                    self._topk_reviews, self.item_ids)
    
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
    
    def get_current_recommended_items(self):
        return self._current_recommended_items 

    def _get_prompt_to_format_recommendation(self):
        """
        Get the prompt to get recommendation text with explanation.
        :param explanation: explanation of why the items are recommended
        :return: prompt to get recommendation text with explanation
        """
        item_names = ' and '.join(
            [f'{rec_item.get_name()}' for rec_item in self._current_recommended_items])
        explanation_str = ', '.join(
            [f'{key}: {val}' for key, val in self.explanation.items()])
        return self._format_recommendation_prompt.render(
            item_names=item_names, explanation=explanation_str, domain=self._domain)

    def _get_explanation_for_each_item(self, state_manager: StateManager) -> dict[Any, str]:
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
                prompt = self._get_prompt_to_explain_recommendation(item_name, metadata, reviews,
                                                                    hard_constraints, soft_constraints)

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

                prompt = self._get_prompt_to_explain_recommendation(item_name, metadata, summarized_reviews,
                                                                    hard_constraints, soft_constraints)

                explanation[item_name] = self._llm_wrapper.make_request(
                    prompt)

        self.explanation = explanation
        
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
        """
        constraints_str = ', '.join(
            [f'{key}: {val}' for key, val in constraints.items()])
        return self._summarize_review_prompt.render(constraints=constraints_str, review=review, domain=self._domain)

    def _get_metadata_of_rec_item(self, recommended_item: RecommendedItem):
        """
        Get metadata of an item used for recommend
        :param recommended_item: recommended item whose metadata to be returned
        """
        attributes = ', '.join(
            [f'{key}: {val}' for key, val in recommended_item.get_mandatory_data().items()] +
            [f'{key}: {val}' for key, val in recommended_item.get_optional_data().items()])
        return attributes
    
    def _notify_observers(self) -> None:
        """
        Notify observers that there are some difficulties.
        """
        for observer in self._observers:
            observer.notify_warning()