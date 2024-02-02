from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retriever.embedder.bert_embedder import BERT_model
from information_retriever.embedder.statics import *
from information_retriever.information_retrieval import InformationRetrieval
from intelligence.gpt_wrapper import GPTWrapper
from information_retriever.search_engine.matmul_search_engine import MatMulSearchEngine
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.metadata_wrapper import MetadataWrapper
from information_retriever.item.item_loader import ItemLoader
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from information_retriever.search_engine.vector_database_search_engine import VectorDatabaseSearchEngine
from information_retriever.vector_database import VectorDataBase
import faiss
import pytest
import pandas as pd
import os
import torch
import dotenv

dotenv.load_dotenv()

config = {
    "NUM_REVIEWS_TO_RETURN": 3,
    "BERT_MODEL_NAME": "TASB",
    "ANSWER_PROMPTS_PATH": "prompt_files/recaction_prompts/answer_prompts",
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
embedder = BERT_model(BERT_model_name, tokenizer_name, True)
items_metadata = pd.read_json("test/information_retriever/data/50_restaurants_metadata.json", orient='records', lines=True)
metadata_wrapper = MetadataWrapper(items_metadata)
reviews_df = pd.read_csv("test/information_retriever/data/50_restaurants_reviews.csv")
review_item_ids = reviews_df["item_id"].to_numpy()
reviews = reviews_df["Review"].to_numpy()
database = faiss.read_index("test/information_retriever/data/50_restaurants_database.faiss")
domain_specific_config_loader = DomainSpecificConfigLoader(config)

filter_item = FilterApplier(metadata_wrapper, domain_specific_config_loader.load_filters())
pd_search_engine = MatMulSearchEngine(embedder, review_item_ids, reviews,
                                      torch.load("test/information_retriever/data/50_restaurants_review_embedding_matrix.pt"),
                                      metadata_wrapper)
vector_database_search_engine = VectorDatabaseSearchEngine(embedder, review_item_ids, reviews,
                                                           VectorDataBase(database), metadata_wrapper)

pd_information_retriever = InformationRetrieval(pd_search_engine, metadata_wrapper, ItemLoader())
vector_database_information_retriever = InformationRetrieval(vector_database_search_engine, metadata_wrapper,
                                                             ItemLoader())

llm_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
item_loader = ItemLoader()

dataset_for_testing = pd.read_csv(
    "test/information_retriever/get_best_matching_review_test.csv", encoding="ISO-8859-1")
dataset_for_testing.reset_index(drop=True, inplace=True)

test_data = []
size = dataset_for_testing.shape[0]
for row in range(size):
    datum = (dataset_for_testing['index_of_restaurant'][row],
             dataset_for_testing['question'][row])
    test_data.append(datum)


class TestSearchEngine:

    @pytest.mark.parametrize("index_of_restaurant, question", test_data)
    def test_get_best_matching_reviews_of_restaurant(self, index_of_restaurant: int, question: str) -> None:
        """
        Test get_best_matching_reviews_of_restaurant() by checking whether it can retrieve the expected review.

        :param index_of_restaurant: index of restaurant in metadata
        :param question: question by user
        """
        recommended_item = item_loader.create_recommended_item(
            "", metadata_wrapper.get_item_dict_from_index(index_of_restaurant), [""])
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, filter_item, None,
                                                "restaurant", hard_coded_responses,
                                                domain_specific_config_loader.load_answer_extract_category_fewshots(),
                                                domain_specific_config_loader.load_answer_ir_fewshots(),
                                                domain_specific_config_loader.load_answer_separate_questions_fewshots()
                                                )
        query = answer_resp.convert_state_to_query(question)
        item_index = filter_item.filter_by_current_item(recommended_item)
        pd_reviews = pd_information_retriever.get_best_matching_reviews_of_item(
            query, answer_resp._num_of_reviews_to_return, item_index)
        vector_database_reviews = vector_database_information_retriever.get_best_matching_reviews_of_item(
            query, answer_resp._num_of_reviews_to_return, item_index)

        assert pd_reviews == vector_database_reviews
