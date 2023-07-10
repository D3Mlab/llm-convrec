import pandas as pd
import pytest
import yaml

from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
from intelligence.gpt_wrapper import GPTWrapper
from rec_action.answer import Answer


domain = "restaurants"
test_file_path = 'test/rec_action/qa_category_extraction_test.csv'
test_df = pd.read_csv(test_file_path)
test_data = [
    (
        row['utterance'],
        {key.strip(): "" for key in row['restaurant attribute keys'].split()} if isinstance(row['restaurant attribute keys'], str) else {},
        row['category']
    )
    for row in test_df.to_dict("records")]

class TestAnswer:

    @pytest.fixture(params=[GPTWrapper()])
    def answer(self, request):
        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        yield Answer(config, request.param, None, None, domain)

    @pytest.mark.parametrize('utterance,restaurant_attributes,category', test_data)
    def test_extract_category_from_input(self, answer, utterance, restaurant_attributes, category) -> None:
        dictionary_info = {"name": "name",
                           "address": "address",
                           "city": "city",
                           "state": "state",
                           "postal_code": "postal_code",
                           "latitude": 0,
                           "longitude": 0,
                           "stars": 0,
                           "review_count": 0,
                           "is_open": True,
                           "attributes": restaurant_attributes,
                           "categories": [],
                           "hours": {}}
        restaurant = RecommendedItem(Item("business_id", dictionary_info), "", [])

        actual = answer._extract_category_from_input(utterance, restaurant)
        assert actual == category
