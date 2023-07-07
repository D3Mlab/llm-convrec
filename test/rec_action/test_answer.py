import pandas as pd
import pytest
import yaml

from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
from intelligence.gpt_wrapper import GPTWrapper
from rec_action.answer import Answer
from state.common_state_manager import CommonStateManager
from state.message import Message

test_file_path = 'rec_action/qa_category_extraction_test.csv'
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
        with open("../config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        yield Answer(config, request.param, None, "restaurants")

    @pytest.mark.parametrize('utterance,restaurant_attributes,category', test_data)
    def test_extract_category_from_input(self, answer, utterance, restaurant_attributes, category) -> None:
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', utterance))
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

        actual = answer._extract_category_from_input(state_manager, restaurant)
        assert actual == category
