import pandas as pd
import pytest

from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
from intelligence.gpt_wrapper import GPTWrapper
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from state.common_state_manager import CommonStateManager
from state.message import Message

domain = "appliances"
test_file_path = 'test/current_appliances_extractor_test.csv'
test_df = pd.read_csv(test_file_path)

recommended_restaurants = []
test_data = []

for col in test_df.to_dict("records"):
    row = [col["user_input"]]
    if col["current_restaurant_names"] != "None.":

        list_curr_restaurant_names = col["current_restaurant_names"][:-1].split(
            ',')
        one_turn_reccommended_restaurants = []

        for curr_restaurant_name in list_curr_restaurant_names:
            dictionary_info = {"name": curr_restaurant_name,
                               "address": "address",
                               "city": "city",
                               "state": "state",
                               "postal_code": "postal_code",
                               "latitude": 0,
                               "longitude": 0,
                               "stars": 0,
                               "review_count": 0,
                               "is_open": True,
                               "attributes": {},
                               "categories": [],
                               "hours": {}}
            recommended_restaurant = RecommendedItem(Item("business_id", dictionary_info), "", [])
            one_turn_reccommended_restaurants.append(recommended_restaurant)

        row.append(one_turn_reccommended_restaurants)
        recommended_restaurants.append(one_turn_reccommended_restaurants)
    else:
        row.append([])

    test_data.append(row)

for row in test_data:
    row.append(recommended_restaurants)


class TestCurrRestaurantsExtractor:

    @pytest.mark.parametrize('user_input,list_curr_restaurant_objs,recommended_restaurants', tuple(test_data))
    def test_extract_category_from_input(self, user_input, list_curr_restaurant_objs, recommended_restaurants) -> None:
        gpt_wrapper = GPTWrapper()

        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', user_input))

        conv_history = state_manager.get("conv_history")

        extractor = CurrentItemsExtractor(gpt_wrapper, domain)

        answer = extractor.extract(recommended_restaurants, conv_history)
        assert list_curr_restaurant_objs == answer
