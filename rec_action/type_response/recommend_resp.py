from information_retrievers.neural_information_retriever import InformationRetriever
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from intelligence.llm_wrapper import LLMWrapper
from jinja2 import Environment, FileSystemLoader


class RecommendResponse:
    """
    Class representing the recommend rec action response
    """
    def __init__(self, domain: str):
        self._current_recommended_items = []
        self._domain = domain
        
    def get_current_recommended_items(self):
        return self._current_recommended_items