import logging
from jinja2 import Environment, FileSystemLoader, Template

from state.state_manager import StateManager
from rec_action.response_type.hard_coded_based_resp import HardCodedBasedResponse
from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.information_retrieval import InformationRetrieval
from intelligence.llm_wrapper import LLMWrapper
from rec_action.response_type.recommend_resp import RecommendResponse

logger = logging.getLogger('recommend')


class RecommendHardCodedBasedResponse(RecommendResponse, HardCodedBasedResponse):
    """
    Class representing the hard coded based response for recommend
    """
    _llm_wrapper: LLMWrapper
    _filter_restaurants: FilterApplier
    _information_retriever: InformationRetrieval
    _topk_items: int
    _topk_reviews: int
    _convert_state_to_query_prompt: Template

    def __init__(self, llm_wrapper: LLMWrapper, filter_restaurants: FilterApplier,
                 information_retriever: InformationRetrieval, domain: str, config: dict, hard_coded_reponses: list[dict]):
        
        super().__init__(domain)
        
        self._filter_restaurants = filter_restaurants
        self._information_retriever = information_retriever
        self._llm_wrapper = llm_wrapper
                
        self._topk_items = int(config["TOPK_ITEMS"])
        self._topk_reviews = int(config["TOPK_REVIEWS"])
        
        env = Environment(loader=FileSystemLoader(
            config['RECOMMEND_PROMPTS_PATH']))
        
        self._convert_state_to_query_prompt = env.get_template(
            config['CONVERT_STATE_TO_QUERY_PROMPT_FILENAME'])
      
    def get(self, state_manager: StateManager) -> str:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """
        # TODO: make it work for newest filter
        query = self.convert_state_to_query(state_manager)

        logger.debug(f'Query: {query}')

        filtered_embedding_matrix = \
            self._filter_restaurants.filter_by_constraints(state_manager)
        for response_dict in self._hard_coded_responses:
            if response_dict['action'] == 'NoRecommendation':
                no_recom_response = response_dict['response']

        try:
            self._current_recommended_items = \
                self._information_retriever.get_best_matching_items(query, self._topk_items,
                                                                    self._topk_reviews, filtered_embedding_matrix)
            return self._format_hard_coded_resp(self._current_recommended_items)
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            return no_recom_response

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

    def _format_hard_coded_resp(self, recommended_items: list[RecommendedItem]) -> str:
        """
        Formats the hard coded recommender response.
        Returns the generated string from the list of recommended items.

        :param recommended_items: list of recommended items

        :return: response to display to user
        """

        recomm_resp = "How about "
        for group in recommended_items:
            for item in group:
                recomm_resp += f'{item.get_name()} or '

        return f'{recomm_resp[:-4]}?'
    
