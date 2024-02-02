from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from rec_action.response_type.recommend_prompt_based_resp import RecommendPromptBasedResponse
from state.common_state_manager import CommonStateManager
from information_retriever.filter.word_in_filter import WordInFilter
from information_retriever.search_engine.matmul_search_engine import MatMulSearchEngine
from information_retriever.item.item_loader import ItemLoader
from information_retriever.embedder.bert_embedder import BERT_model
from information_retriever.embedder.statics import *
from information_retriever.information_retrieval import InformationRetrieval
from domain_specific.classes.restaurants.location_filter import LocationFilter
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.item.recommended_item import RecommendedItem
from information_retriever.metadata_wrapper import MetadataWrapper
from intelligence.gpt_wrapper import GPTWrapper
from information_retriever.search_engine.vector_database_search_engine import VectorDatabaseSearchEngine
from information_retriever.vector_database import VectorDataBase
from information_retriever.search_engine.search_engine import SearchEngine
import pytest
import pandas as pd
from dotenv import load_dotenv
import os
import torch
import faiss

load_dotenv()

def fill_in_list():
    df = pd.read_csv("test/information_retriever/get_best_matching_item_test.csv", encoding = "ISO-8859-1")
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
    "TOPK_ITEMS": 3,
    "TOPK_REVIEWS": 5,
    "BERT_MODEL_NAME": "TASB",
    "RECOMMEND_PROMPTS_PATH": "prompt_files/recaction_prompts/recommend_prompts",
    "CONVERT_STATE_TO_QUERY_PROMPT_FILENAME": "convert_state_to_query_prompt.jinja",
    "EXPLAIN_RECOMMENDATION_PROMPT_FILENAME": "explain_recommendation_prompt.jinja",
    "FORMAT_RECOMMENDATION_PROMPT_FILENAME": "format_recommendation_prompt.jinja",
    "NO_MATCHING_ITEM_PROMPT_FILENAME": "no_matching_item_prompt.jinja",
    "SUMMARIZE_REVIEW_PROMPT_FILENAME": "summarize_review_prompt.jinja",
    "RECOMMEND_RESPONSE_TYPE": "prompt",
    'PREFERENCE_ELICITATION_PROMPT': "preference_elicitation_prompt.jinja",
    'ENABLE_PREFERENCE_ELICITATION': False,
    "ENABLE_MULTITHREADING": False
}

word_in_filter = WordInFilter(["cuisine type", "dish type"], "categories")
location_filter = LocationFilter("location", ["latitude", "longitude"], 2, GoogleV3Wrapper())
items_metadata = pd.read_json("test/information_retriever/data/50_restaurants_metadata.json", orient='records', lines=True)
metadata_wrapper = MetadataWrapper(items_metadata)
filter_item = FilterApplier(metadata_wrapper, [word_in_filter, location_filter])
BERT_name = config["BERT_MODEL_NAME"]
BERT_model_name = BERT_MODELS[BERT_name]
tokenizer_name = TOEKNIZER_MODELS[BERT_name]
embedder = BERT_model(BERT_model_name, tokenizer_name, True)

reviews_df = pd.read_csv("test/information_retriever/data/50_restaurants_reviews.csv")
review_item_ids = reviews_df["item_id"].to_numpy()
reviews = reviews_df["Review"].to_numpy()
reviews_embedding_matrix = torch.load("test/information_retriever/data/50_restaurants_review_embedding_matrix.pt")
database = faiss.read_index("test/information_retriever/data/50_restaurants_database.faiss")

pd_search_engine = MatMulSearchEngine(embedder, review_item_ids, reviews, reviews_embedding_matrix, metadata_wrapper)
vector_database_search_engine = VectorDatabaseSearchEngine(embedder, review_item_ids, reviews,
                                                           VectorDataBase(database), metadata_wrapper)
test_data = fill_in_list()


class TestGetBestMatchingItems:

    @pytest.mark.parametrize("state_manager, expected_item_name", test_data)
    @pytest.mark.parametrize("should_filter, search_engine",
                             [(False, pd_search_engine),
                              (False, vector_database_search_engine),
                              (True, pd_search_engine),
                              (True, vector_database_search_engine)])
    def test_get_best_matching_items(self, state_manager: CommonStateManager,
                                     expected_item_name: str, should_filter: bool,
                                     search_engine: SearchEngine) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param state_manager: state_manager to be converted to query
        :param expected_item_name: item name that the function is supposed to return
        """
        information_retriever = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())
        llm_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
        recommend = RecommendPromptBasedResponse(
            llm_wrapper, filter_item, information_retriever, "restaurants", [], config, [])
        query = recommend.convert_state_to_query(state_manager)
        if should_filter:
            item_indices = filter_item.apply_filter(state_manager)
        else:
            item_indices = metadata_wrapper.get_metadata().index.tolist()

        recommended_items = information_retriever.get_best_matching_items(query, config['TOPK_ITEMS'],
                                                                          config['TOPK_REVIEWS'],
                                                                          item_indices)

        recommended_items_flattened = [group[0] for group in recommended_items if len(group) != 0]
        item_names = self._create_item_name_list_from_recommended_item_list(recommended_items_flattened)
        assert expected_item_name in item_names

    def _create_item_name_list_from_recommended_item_list(
            self, recommended_items: list[RecommendedItem]) -> list[str]:
        """
        Create a list of restaurant names from a list of RecommendedIem.

        :param recommended_items: list of recommended items
        """
        item_names = []
        for rec_item in recommended_items:
            item_names.append(rec_item.get_name())
        return item_names
