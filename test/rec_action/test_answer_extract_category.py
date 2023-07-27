import pandas as pd
import pytest
import yaml
import os
from dotenv import load_dotenv
from intelligence.gpt_wrapper import GPTWrapper
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from information_retrievers.item.item_loader import ItemLoader


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


class TestAnswerExtractCategory:

    @pytest.mark.parametrize('utterance, restaurant_attributes, expected_category', test_data1)
    def test_extract_category_from_input_restaurant(self, utterance, restaurant_attributes,
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
        answer_resp = AnswerPromptBasedResponse(config,gpt_wrapper, None,
                                                None, "restaurants", None)
        actual = answer_resp._extract_category_from_input(utterance, restaurant)
        assert actual == expected_category

    @pytest.mark.parametrize('utterance, clothing_attributes, expected_category', test_data2)
    def test_extract_category_from_input_clothing(self, utterance, clothing_attributes,
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
        answer_resp = AnswerPromptBasedResponse(config, gpt_wrapper, None,
                                                None, "clothing", None)
        actual = answer_resp._extract_category_from_input(utterance, clothing)
        assert actual == expected_category
