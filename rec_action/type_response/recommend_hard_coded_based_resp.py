from state.state_manager import StateManager

from rec_action.type_response.hard_coded_based_resp import HardCodedBasedResponse
from state.state_manager import StateManager
from information_retrievers.neural_information_retriever import InformationRetriever
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from intelligence.llm_wrapper import LLMWrapper
import logging
from jinja2 import Environment, FileSystemLoader


from rec_action.type_response.recommend_resp import RecommendResponse

logger = logging.getLogger('recommend')

class ReccomendHardCodedBasedResponse(RecommendResponse, HardCodedBasedResponse):
    """
    Class representing the hard coded based response for recommend
    """
    _llm_wrapper: LLMWrapper
    _filter_restaurants: FilterRestaurants
    _information_retriever: InformationRetriever
    _topk_items: str
    _topk_reviews: str
    _convert_state_to_query_prompt: str

    def __init__(self, llm_wrapper: LLMWrapper, filter_restaurants: FilterRestaurants,
                 information_retriever: InformationRetriever, domain: str, config: dict):
        
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
      
    def get_response(self, state_manager: StateManager) -> str:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """

        query = self.convert_state_to_query(state_manager)

        logger.debug(f'Query: {query}')

        filtered_embedding_matrix = \
            self._filter_restaurants.filter_by_constraints(state_manager)

        try:
            self._current_recommended_items = \
                self._information_retriever.get_best_matching_items(query, self._topk_items,
                                                                    self._topk_reviews, filtered_embedding_matrix)
            return self._format_hard_coded_resp(self._current_recommended_items)
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            return f"Sorry, there is no {self._domain} that match your constraints."

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

    #TODO: generalize this once metadata is done
    def _format_hard_coded_resp(self, recommended_restaurants: list[RecommendedItem]) -> str:
        """
        Formats the hard coded recommender response.
        Returns the generated string from the list of recommended restaurants.

        :param recommended_restaurants: list of recommended restaurants

        :return: response to display to user
        """

        recomm_resp = "How about "
        for restaurant in recommended_restaurants:
            recomm_resp += f'{restaurant.get("name")} at {restaurant.get("address")}, {restaurant.get("city")} with {restaurant.get("stars")} stars out of {restaurant.get("review_count")} reviews or '

        return f'{recomm_resp[:-4]}?'
    
