from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.embedder.statics import *
from information_retrievers.information_retrieval import InformationRetrieval
from intelligence.gpt_wrapper import GPTWrapper
from information_retrievers.search_engine.pd_search_engine import PDSearchEngine
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.metadata_wrapper import MetadataWrapper
from information_retrievers.item.item_loader import ItemLoader
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
import pytest
import pandas as pd
import os
import torch
import dotenv

dotenv.load_dotenv()

config = {
    "NUM_REVIEWS_TO_RETURN": 3,
    "BERT_MODEL_NAME": "TASB",
    "ANSWER_PROMPTS_PATH": "prompt_files/gpt/recaction_prompts/answer_prompts",
    "ANSWER_VERIFY_METADATA_RESP_PROMPT": "verify_metadata_resp_prompt.jinja",
    "ANSWER_FORMAT_MULTIPLE_RESP_PROMPT": "format_multiple_resp_prompt.jinja",
    "ANSWER_EXTRACT_CATEGORY_PROMPT": "extract_category_prompt.jinja",
    "ANSWER_HOURS_PROMPT": "hours_prompt.jinja",
    "ANSWER_METADATA_PROMPT": "metadata_prompt.jinja",
    "ANSWER_IR_PROMPT": "ir_prompt.jinja",
    "ANSWER_MULT_QS_PROMPT": "seperate_questions_prompt.jinja",
    "ANSWER_GPT_PROMPT": "gpt_prompt.jinja",
    "ANSWER_MULT_QS_FORMAT_RESP_PROMPT": "format_mult_qs_prompt.jinja",
    "ENABLE_MULTITHREADING": False,
    'PATH_TO_DOMAIN_CONFIGS': 'domain_specific/configs/restaurant_configs'
}

hard_coded_responses = [
    {'action': 'NoAnswer',
     'response': 'Please only ask questions about previously recommended restaurant.',
     'constraints': []}
]
BERT_name = config["BERT_MODEL_NAME"]
BERT_model_name = BERT_MODELS[BERT_name]
tokenizer_name = TOEKNIZER_MODELS[BERT_name]
embedder = BERT_model(BERT_model_name, tokenizer_name, False)
items_metadata = pd.read_json("test/information_retriever/data/Edmonton_restaurants.json", orient='records', lines=True)
metadata_wrapper = MetadataWrapper(items_metadata)
reviews_df = pd.read_csv("test/information_retriever/data/Edmonton_restaurants_review.csv")
domain_specific_config_loader = DomainSpecificConfigLoader(config)

filter_item = FilterApplier(metadata_wrapper, domain_specific_config_loader.load_filters())
search_engine = PDSearchEngine(embedder, reviews_df["item_id"].to_numpy(), reviews_df["Review"].to_numpy(),
                               torch.load("test/information_retriever/data/matrix.pt"))
information_retriever = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())
llm_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])

item_loader = ItemLoader()

dataset_for_testing = pd.read_csv(
    "test/information_retriever/get_best_matching_review_test.csv", encoding="ISO-8859-1")
dataset_for_testing.reset_index(drop=True, inplace=True)

test_data = []
size = dataset_for_testing.shape[0]
for row in range(size):
    datum = (dataset_for_testing['index_of_restaurant'][row],
             dataset_for_testing['question'][row], dataset_for_testing['expected_review'][row])
    test_data.append(datum)


class TestGetBestMatchingReviewsOfRestaurant:

    @pytest.mark.parametrize("index_of_restaurant, question, expected_review", test_data)
    def test_get_best_matching_reviews_of_restaurant(self, index_of_restaurant: int, question: str,
                                                     expected_review: str) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param index_of_restaurant: index of restaurant in metadata
        :param question: question by user
        :param expected_review: review that the function is supposed to return 
        """
        recommended_item = item_loader.create_recommended_item(
            "", metadata_wrapper.get_item_dict_from_index(index_of_restaurant), [""])
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, filter_item, information_retriever,
                                                "restaurant", hard_coded_responses,
                                                domain_specific_config_loader.load_answer_extract_category_fewshots(),
                                                domain_specific_config_loader.load_answer_ir_fewshots(),
                                                domain_specific_config_loader.load_answer_separate_questions_fewshots(),
                                                domain_specific_config_loader.load_answer_verify_metadata_resp_fewshots(),
                                                )
        query = answer_resp.convert_state_to_query(question)
        item_index = filter_item.filter_by_current_item(recommended_item)
        reviews = information_retriever.get_best_matching_reviews_of_item(
            query, answer_resp._num_of_reviews_to_return, item_index)
        retrieved_review = [group[0] for group in reviews]

        retrieved_review_stripped = []
        for review in retrieved_review[0]:
            retrieved_review_stripped.append(review.replace(" ", "").replace("\r", "").replace("\n", ""))
        expected_review = expected_review.replace(" ", "").replace("\r", "").replace("\n", "")
        assert expected_review in retrieved_review_stripped
