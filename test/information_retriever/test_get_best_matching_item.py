from rec_action.response_type.recommend_prompt_based_resp import RecommendPromptBasedResponse
from state.common_state_manager import CommonStateManager
from information_retrievers.filter.word_in_filter import WordInFilter
from information_retrievers.search_engine.pd_search_engine import PDSearchEngine
from information_retrievers.item.item_loader import ItemLoader
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.embedder.statics import *
from information_retrievers.information_retrieval import InformationRetrieval
from domain_specific.classes.restaurants.location_filter import LocationFilter
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.metadata_wrapper import MetadataWrapper
from intelligence.gpt_wrapper import GPTWrapper
import pytest
import pandas as pd
from dotenv import load_dotenv
import os
import torch

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
    "RECOMMEND_PROMPTS_PATH": "prompt_files/gpt/recaction_prompts/recommend_prompts",
    "CONVERT_STATE_TO_QUERY_PROMPT_FILENAME": "convert_state_to_query_prompt.jinja",
    "EXPLAIN_RECOMMENDATION_PROMPT_FILENAME": "explain_recommendation_prompt.jinja",
    "FORMAT_RECOMMENDATION_PROMPT_FILENAME": "format_recommendation_prompt.jinja",
    "NO_MATCHING_ITEM_PROMPT_FILENAME": "no_matching_item_prompt.jinja",
    "SUMMARIZE_REVIEW_PROMPT_FILENAME": "summarize_review_prompt.jinja",
    "RECOMMEND_RESPONSE_TYPE": "prompt",
    "ENABLE_MULTITHREADING": False
}

word_in_filter = WordInFilter(["cuisine type, dish type"], "categories")
location_filter = LocationFilter("location", ["latitude", "longitude"], 2)
metadata_wrapper = MetadataWrapper()
metadata_wrapper.items_metadata = pd.read_json("test/information_retriever/data/Edmonton_restaurants.json", orient='records', lines=True)
filter_item = FilterApplier(metadata_wrapper)
filter_item.filters = [word_in_filter, location_filter]
BERT_name = config["BERT_MODEL_NAME"]
BERT_model_name = BERT_MODELS[BERT_name]
tokenizer_name = TOEKNIZER_MODELS[BERT_name]
embedder = BERT_model(BERT_model_name, tokenizer_name, False)
search_engine = PDSearchEngine(embedder)
reviews_df = pd.read_csv("test/information_retriever/data/Edmonton_restaurants_review.csv")
search_engine._review_item_ids = reviews_df["item_id"].to_numpy()
search_engine._reviews = reviews_df["Review"].to_numpy()
search_engine._reviews_embedding_matrix = torch.load("test/information_retriever/data/matrix.pt")
information_retriever = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())
test_data = fill_in_list()


class TestGetBestMatchingItems:

    @pytest.mark.parametrize("state_manager, expected_item_name", test_data)
    @pytest.mark.parametrize("should_filter", [False, True])
    def test_get_best_matching_items(self, state_manager: CommonStateManager,
                                     expected_item_name: str, should_filter: bool) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param state_manager: state_manager to be converted to query
        :param expected_item_name: item name that the function is supposed to return
        """
        llm_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
        recommend = RecommendPromptBasedResponse(
            llm_wrapper, filter_item, information_retriever, "restaurants", [], config)
        query = recommend.convert_state_to_query(state_manager)
        if should_filter:
            item_indices = filter_item.apply_filter(state_manager)
        else:
            item_indices = metadata_wrapper.get_metadata().index.tolist()

        recommended_items = information_retriever.get_best_matching_items(query, config['TOPK_ITEMS'],
                                                                          config['TOPK_REVIEWS'],
                                                                          item_indices)
        item_names = self._create_item_name_list_from_recommended_item_list(recommended_items)
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