from rec_action.answer import Answer
from state.common_state_manager import CommonStateManager
from user_intent.ask_for_recommendation import AskForRecommendation
from information_retrievers.filter.check_location import CheckLocation
from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.neural_ir.neural_embedder import BERT_model
from information_retrievers.neural_ir.statics import *
from information_retrievers.neural_ir.neural_search_engine import NeuralSearchEngine
from information_retrievers.neural_information_retriever import NeuralInformationRetriever
from intelligence.llm_wrapper import LLMWrapper
from information_retrievers.item import Item
from information_retrievers.recommended_item import RecommendedItem
from user_intent.inquire import Inquire
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from state.message import Message
from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from domain_specific.classes.restaurants.geocoding.nominatim_wrapper import NominatimWrapper

from information_retrievers.data_holder import DataHolder
import pytest
import pandas as pd
import ast

config = {
    "TOPK_RESTAURANTS": 2,
    "TOPK_REVIEWS": 2,
    "BERT_MODEL_NAME": "TASB",
    "PATH_TO_RESTAURANT_METADATA": "information_retrievers/data/Top-50-Restaurants/top_50_restaurants_sorted.csv",
    "PATH_TO_RESTAURANT_REVIEW_EMBEDDINGS": "information_retrievers/data/Top-50-Restaurants/top_50_restaurants_review_embedding_sorted.csv",
    "PATH_TO_RESTAURANT_REVIEW_EMBEDDING_MATRIX": "information_retrievers/data/Top-50-Restaurants/matrix.pt",
    "PATH_TO_NUM_OF_REVIEWS_PER_RESTAURANT": "information_retrievers/data/Top-50-Restaurants/item.pt",
    "DEFAULT_MAX_DISTANCE_IN_KM": 2,
    "DISTANCE_TYPE": "geodesic",
    "FILTER_CONSTRAINTS": "location",
    "NUM_REVIEWS_TO_RETURN": 3
}
data_holder = DataHolder("information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_sorted.csv",
                         "information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_review_embedding_sorted.csv",
                         "information_retrievers/data/Edmonton-Restaurants/matrix.pt",
                         "information_retrievers/data/Edmonton-Restaurants/item.pt")
BERT_name = config["BERT_MODEL_NAME"]
BERT_model_name = BERT_MODELS[BERT_name]
tokenizer_name = TOEKNIZER_MODELS[BERT_name]
embedder = BERT_model(BERT_model_name, tokenizer_name, False)
engine = NeuralSearchEngine(embedder)
geocoder_wrapper = NominatimWrapper()
check_location = CheckLocation(config['DEFAULT_MAX_DISTANCE_IN_KM'], "geodesic")
check_cuisine_type = CheckCuisineDishType()
check_already_recommended_restaurant = CheckAlreadyRecommendedRestaurant()
filter_restaurant = FilterRestaurants(geocoder_wrapper, check_location, check_cuisine_type,
                                      check_already_recommended_restaurant,
                                      data_holder, config["FILTER_CONSTRAINTS"])
information_retriever = NeuralInformationRetriever(engine, data_holder)
llm_wrapper = LLMWrapper()
curr_restaurant_extractor = CurrentItemsExtractor(llm_wrapper)
user_intents = [AskForRecommendation(), Inquire(curr_restaurant_extractor)]
state_manager = CommonStateManager(
    {user_intents[0], user_intents[1]}, user_intents[0])
answer_rec_action = Answer(config, llm_wrapper, filter_restaurant, information_retriever, "restaurants")
restaurant_meta_data = pd.read_csv(
    "information_retrievers/data/Edmonton-Restaurants/Edmonton_restaurants_sorted.csv", encoding = "ISO-8859-1")

dataset_for_testing = pd.read_csv(
    "test/information_retriever/neural_ir_get_review_top50_test.csv", encoding = "ISO-8859-1")
dataset_for_testing.reset_index(drop=True, inplace=True)

test_data = []
size = dataset_for_testing.shape[0]
for row in range(size):
    datum = (dataset_for_testing['index_of_restaurant'][row],
             dataset_for_testing['question'][row], dataset_for_testing['expected_review'][row])
    test_data.append(datum)

@pytest.mark.parametrize("num_of_reviews_to_return", [3])
class TestGetBestMatchingReviewsOfRestaurant:

    @pytest.mark.parametrize("index_of_restaurant, question, expected_review", test_data)
    def test_get_best_matching_reviews_of_restaurant(self, num_of_reviews_to_return: int,
                                                     index_of_restaurant: int, question: str,
                                                     expected_review: str) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param index_of_restaurant: index of restaurant in restaurant_meta_data 
        :param question: question by user
        :param expected_review: review that the function is supposed to return 
        """
        recommended_restaurant = self._create_recommended_restaurant_obj_from_meta_data(index_of_restaurant,
                                                                                        restaurant_meta_data)
        query = answer_rec_action.convert_state_to_query(
            question, recommended_restaurant)
        filtered_embedding_matrix, filtered_num_of_reviews_per_restaurant, \
            filtered_restaurants_review_embeddings = \
            filter_restaurant.filter_by_restaurant_name([recommended_restaurant.get_name()])
        retrieved_review = information_retriever.get_best_matching_reviews_of_item(
            query, [recommended_restaurant.get_name()], num_of_reviews_to_return,
            filtered_restaurants_review_embeddings,
            filtered_embedding_matrix, filtered_num_of_reviews_per_restaurant)

        retrieved_review_stripped = []
        for review in retrieved_review[0]:
            retrieved_review_stripped.append(review.replace(" ", "").replace("\r", "").replace("\n", ""))
        expected_review = expected_review.replace(" ", "").replace("\r", "").replace("\n", "")
        assert expected_review in retrieved_review_stripped

    def _create_recommended_restaurant_obj_from_meta_data(self, index_of_restaurant: int,
                                                          restaurant_meta_data: pd.DataFrame) -> RecommendedItem:
        restaurant_obj = self._create_restaurant_obj_from_meta_data(
            index_of_restaurant, restaurant_meta_data)
        query = ""
        most_relavent_reviews = [""]
        return RecommendedItem(restaurant_obj, query, most_relavent_reviews)

    def _create_restaurant_obj_from_meta_data(self, index_of_restaurant: int,
                                              restaurant_meta_data: pd.DataFrame) -> Item:
        business_id = restaurant_meta_data.iloc[index_of_restaurant]["business_id"]
        item_info = restaurant_meta_data.loc[restaurant_meta_data['business_id']
                                                   == business_id].iloc[0]
        dictionary_info = {"name": item_info[1],
                           "address": item_info[2],
                           "city": item_info[3],
                           "state": item_info[4],
                           "postal_code": item_info[5],
                           "latitude": float(item_info[6]),
                           "longitude": float(item_info[7]),
                           "stars": float(item_info[8]),
                           "review_count": int(item_info[9]),
                           "is_open": bool(item_info[10]),
                           "attributes": ast.literal_eval(item_info[11]),
                           "categories": list(item_info[12].split(",")),
                           "hours": {}}
        restaurant_object = Item(business_id, dictionary_info)
        return restaurant_object

