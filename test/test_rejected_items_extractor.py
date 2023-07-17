import pytest
import pandas as pd

from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
from intelligence.gpt_wrapper import GPTWrapper
from state.message import Message
from user_intent.extractors.rejected_items_extractor import RejectedItemsExtractor

test_file_path = 'test/rejected_restaurants_extractor_test.csv'
test_df = pd.read_csv(test_file_path, encoding='latin1')
test_data = [
    (
        row["utterance"],
        list(map(lambda x: x.strip(), row['all mentioned items'].split(',')))
        if isinstance(row['all mentioned items'], str) else [],
        list(map(lambda x: x.strip(), row['recently mentioned items'].split(',')))
        if isinstance(row['recently mentioned items'], str) else [],
        list(map(lambda x: x.strip(), row['rejected items'].split(',')))
        if isinstance(row['rejected items'], str) else [],
    )
    for row in test_df.to_dict("records")
]


class TestRejectedRestaurantsExtractorTest:

    @pytest.fixture(params=[GPTWrapper()])
    def rejected_restaurants_extractor(self, request):
        yield RejectedItemsExtractor(request.param, "restaurants")

    @pytest.mark.parametrize("utterance,all_mentioned_restaurant_names,recently_mentioned_restaurant_names,rejected_restaurant_names", test_data)
    def test_extract(self, rejected_restaurants_extractor, utterance, all_mentioned_restaurant_names, recently_mentioned_restaurant_names, rejected_restaurant_names):

        _recently_mentioned_restaurant_names = set(recently_mentioned_restaurant_names)
        _rejected_restaurant_names = set(rejected_restaurant_names)
        all_mentioned_restaurants = [RecommendedItem(Item("item_id", {"name": name}), "", []) for name in all_mentioned_restaurant_names]
        recently_mentioned_restaurants = [restaurant for restaurant in all_mentioned_restaurants if restaurant.get_name() in _recently_mentioned_restaurant_names]
        rejected_restaurants = [restaurant.get_name() for restaurant in all_mentioned_restaurants if restaurant.get_name() in _rejected_restaurant_names]

        conv_history = [Message("user", utterance)]

        actual = rejected_restaurants_extractor.extract(conv_history, all_mentioned_restaurants, recently_mentioned_restaurants)

        assert [restaurant.get_name() for restaurant in actual] == rejected_restaurants
