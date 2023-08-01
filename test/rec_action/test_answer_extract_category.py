import pandas as pd
import pytest
import yaml
import os
from dotenv import load_dotenv

from domain_specific_config_loader import DomainSpecificConfigLoader
from intelligence.gpt_wrapper import GPTWrapper
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from information_retriever.item.item_loader import ItemLoader
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper

load_dotenv()


def load_test_data(domain: str) -> list[(str, str)]:
    """
    Load test data using the domain name.

    :param domain: domain name
    :return: test data which is a list od tuple whose first element has an utterance,
             the second element has item attribute keys, and the third element has the expected category.
    """
    test_file_path = f'test/rec_action/qa_category_extraction_{domain}_test.csv'
    test_df = pd.read_csv(test_file_path)
    test_data = [
        (
            row['utterance'],
            {key.strip(): "" for key in row['item attribute keys'].split(",")} if isinstance(row['item attribute keys'],
                                                                                          str) else {},
            row['expected category']
        )
        for row in test_df.to_dict("records")]
    return test_data


domain1 = "restaurants"
test_data1 = load_test_data(domain1)

domain2 = "clothing"
test_data2 = load_test_data(domain2)

with open("system_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
gpt_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
item_loader = ItemLoader()


@pytest.mark.parametrize('llm_wrapper', [GPTWrapper(os.environ['OPENAI_API_KEY']), AlpacaLoraWrapper(os.environ['GRADIO_URL'])])
class TestAnswerExtractCategory:

    @pytest.mark.parametrize('utterance, restaurant_attributes, expected_category', test_data1)
    def test_extract_category_from_input_restaurant(self, llm_wrapper, utterance, restaurant_attributes,
                                                    expected_category) -> None:
        dictionary_info = {"item_id": "id",
                           "name": "name",
                           "address": "address",
                           "city": "city",
                           "state": "state",
                           "postal_code": "postal_code",
                           "latitude": 0,
                           "longitude": 0,
                           "stars": 0,
                           "review_count": 0,
                           "is_open": True,
                           "optional": restaurant_attributes,
                           "categories": [],
                           "hours": {}}
        restaurant = item_loader.create_recommended_item("", dictionary_info, [""])
        config['PATH_TO_DOMAIN_CONFIGS'] = "domain_specific/configs/restaurant_configs"
        domain_specific_config_loader = DomainSpecificConfigLoader(config)
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, None,
                                                None, "restaurants", None,
                                                domain_specific_config_loader.load_answer_extract_category_fewshots(),
                                                domain_specific_config_loader.load_answer_ir_fewshots(),
                                                domain_specific_config_loader.load_answer_separate_questions_fewshots(),
                                                )
        actual = answer_resp._extract_category_from_input(utterance, restaurant)
        assert actual == expected_category

    @pytest.mark.parametrize('utterance, clothing_attributes, expected_category', test_data2)
    def test_extract_category_from_input_clothing(self, llm_wrapper, utterance, clothing_attributes,
                                                  expected_category) -> None:
        dictionary_info = {"item_id": "id",
                           "name": "name",
                           "category": "category",
                           "price": "price",
                           "brand": "brand",
                           "rating": 0,
                           "num_reviews": 0,
                           "rank": 0,
                           "optional": clothing_attributes}
        clothing = item_loader.create_recommended_item("", dictionary_info, [""])
        config['PATH_TO_DOMAIN_CONFIGS'] = "domain_specific/configs/clothing_configs"
        domain_specific_config_loader = DomainSpecificConfigLoader(config)
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, None,
                                                None, "clothing", None,
                                                domain_specific_config_loader.load_answer_extract_category_fewshots(),
                                                domain_specific_config_loader.load_answer_ir_fewshots(),
                                                domain_specific_config_loader.load_answer_separate_questions_fewshots()                                                )
        actual = answer_resp._extract_category_from_input(utterance, clothing)
        assert actual == expected_category
