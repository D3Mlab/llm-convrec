from rec_action.recommend import Recommend
from state.common_state_manager import CommonStateManager
from information_retrievers.filter.check_location import CheckLocation
from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.neural_ir.neural_embedder import BERT_model
from information_retrievers.neural_ir.statics import *
from information_retrievers.neural_ir.neural_search_engine import NeuralSearchEngine
from information_retrievers.neural_information_retriever import NeuralInformationRetriever
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from state.common_state_manager import StateManager
from geocoding.google_v3_wrapper import GoogleV3Wrapper
from geocoding.nominatim_wrapper import NominatimWrapper
from information_retrievers.data_holder import DataHolder
from intelligence.gpt_wrapper import GPTWrapper
import yaml
import pytest
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fill_in_list():
    df = pd.read_csv("test/information_retriever/neural_ir_get_restaurant_edmonton_test.csv", encoding = "ISO-8859-1")
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    test_data = []

    for i in range(num_rows):
        large_dictionary = {}
        small_dictionary = {}
        expected_restaurant_name = ""
        for j in range(num_columns):
            if (pd.isna(df.iloc[i, j])):
                continue
            else:
                if (df.columns[j] == "expected_restaurant_name"):
                    expected_restaurant_name = df.iloc[i, j]
                else:
                    small_dictionary[df.columns[j]] = df.iloc[i, j].split(" # ")

        large_dictionary["hard_constraints"] = small_dictionary
        state = CommonStateManager({}, data=large_dictionary)
        test_data.append((state, expected_restaurant_name))

    return test_data


config = {
    "TOPK_RESTAURANTS": 1,
    "TOPK_REVIEWS": 5,
    "BERT_MODEL_NAME": "TASB",
    "PATH_TO_RESTAURANT_METADATA": "information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_sorted.csv",
    "PATH_TO_RESTAURANT_REVIEW_EMBEDDINGS": "information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_review_embedding_sorted.csv",
    "PATH_TO_RESTAURANT_REVIEW_EMBEDDING_MATRIX": "information_retrievers/data/Edmonton-Restaurants/matrix.pt",
    "PATH_TO_NUM_OF_REVIEWS_PER_RESTAURANT": "information_retrievers/data/Edmonton-Restaurants/item.pt",
    "DEFAULT_MAX_DISTANCE_IN_KM": 2,
    "DISTANCE_TYPE": "geodesic",
    "FILTER_CONSTRAINTS": ["location", "cuisine dish type"]
}
data_holder = DataHolder("information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_sorted.csv",
                         "information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_review_embedding_sorted.csv",
                         "information_retrievers/data/Edmonton-Restaurants/matrix.pt",
                         "information_retrievers/data/Edmonton-Restaurants/item.pt")
geocoder_wrapper = GoogleV3Wrapper()
check_location = CheckLocation(config['DEFAULT_MAX_DISTANCE_IN_KM'], "geodesic")
check_cuisine_type = CheckCuisineDishType()
check_already_recommended_restaurant = CheckAlreadyRecommendedRestaurant()
filter_restaurant = FilterRestaurants(geocoder_wrapper, check_location, check_cuisine_type, check_already_recommended_restaurant,
                                      data_holder, config["FILTER_CONSTRAINTS"])
BERT_name = config["BERT_MODEL_NAME"]
BERT_model_name = BERT_MODELS[BERT_name]
tokenizer_name = TOEKNIZER_MODELS[BERT_name]
embedder = BERT_model(BERT_model_name, tokenizer_name, False)
engine = NeuralSearchEngine(embedder)
information_retriever = NeuralInformationRetriever(engine, data_holder)
test_data = fill_in_list()



class TestGetBestMatchingRestaurants:

    @pytest.mark.parametrize("state_manager, expected_restaurant_name", test_data)
    @pytest.mark.parametrize("topk_restaurants, topk_reviews, should_filter", [(3, 5, False), (3, 5, True)])
    def test_get_best_matching_restaurants(self, state_manager: CommonStateManager,
                                           expected_restaurant_name: str, topk_restaurants: int,
                                           topk_reviews: int, should_filter: bool) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param state_manager: state_manager to be converted to query
        :param expected_restaurant_name: restaurant name that the function is supposed to return 
        :param topk_restaurants: the number of restaurants to return
        :param topk_reviews: the number of reviews to store in a RecommendedRestaurant object
        """
        llm_wrapper = GPTWrapper(observers=[self])
        recommend = Recommend(llm_wrapper, filter_restaurant, information_retriever,
                              mandatory_constraints={"location", "cuisine type"},
                              specific_location_required=False)
        query = recommend.convert_state_to_query(state_manager)
        cuisine_type = state_manager.get("hard_constraints").get("cuisine type")
        dish_type = state_manager.get("hard_constraints").get("dish type")
        print(cuisine_type)
        print(dish_type)
        if should_filter:
            filtered_embedding_matrix = \
                filter_restaurant.filter_by_constraints(state_manager)
        else:
            filtered_embedding_matrix = information_retriever._data_holder.get_item_embedding_matrix()

        recommended_restaurants = information_retriever.get_best_matching_items(query, topk_restaurants,
                                                                                topk_reviews,
                                                                                filtered_embedding_matrix)

        restaurant_names = self._create_restaurant_name_list_from_recommended_estaurant_list(recommended_restaurants)
        assert expected_restaurant_name in restaurant_names

    def _create_restaurant_name_list_from_recommended_estaurant_list(self,
                                                                     recommended_restaurants: list[
                                                                         RecommendedItem]) -> list[str]:
        """
        Create a list of restaurant names from a list of RecommendedRestaurant.
        
        :param recommended_restaurants: list of recommended restaurants
        """
        restaurant_names = []
        for rec_restaurant in recommended_restaurants:
            restaurant_names.append(rec_restaurant.get("name"))
        return restaurant_names
