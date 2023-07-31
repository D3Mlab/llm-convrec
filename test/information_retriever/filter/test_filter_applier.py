from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from information_retriever.filter.word_in_filter import WordInFilter
from information_retriever.filter.exact_word_matching_filter import ExactWordMatchingFilter
from domain_specific.classes.restaurants.location_filter import LocationFilter
from information_retriever.filter.value_range_filter import ValueRangeFilter
from information_retriever.filter.item_filter import ItemFilter
from information_retriever.filter.filter import Filter
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.item.item import Item
from information_retriever.item.recommended_item import RecommendedItem
from state.common_state_manager import CommonStateManager
from information_retriever.metadata_wrapper import MetadataWrapper
import pandas as pd
import pytest
import dotenv

dotenv.load_dotenv()

metadata = pd.read_json("test/information_retriever/filter/50_restaurants_metadata.json", orient='records', lines=True)
metadata_wrapper = MetadataWrapper(metadata)

exact_word_matching_filter = ExactWordMatchingFilter(["cuisine type"], "categories")
word_in_filter = WordInFilter(["cuisine type"], "categories")
item_filter = ItemFilter("recommended_items", "name")
value_range_filter = ValueRangeFilter("rating", "stars")
location_filter = LocationFilter("location", ["latitude", "longitude"], 2, GoogleV3Wrapper())

test_csv = pd.read_csv("test/information_retriever/filter/test_filter_applier.csv", encoding ="ISO-8859-1")
num_rows = test_csv.shape[0]
num_columns = test_csv.shape[1]
test_data = []

for i in range(num_rows):
    large_dictionary = {}
    small_dictionary = {}
    filters_list = []
    for j in range(num_columns):
        if pd.isna(test_csv.iloc[i, j]):
            continue
        else:
            if test_csv.columns[j] == "expected_indices":
                expected_indices = [int(value) for value in test_csv.iloc[i, j].split(",")]
            elif test_csv.columns[j] == "name":
                item_names = test_csv.iloc[i, j].split(",")
            elif test_csv.columns[j] == "type_of_filter":
                filter_names = test_csv.iloc[i, j].split(",")
                for filter_name in filter_names:
                    if filter_name.strip() == "exact word matching":
                        filters_list.append(exact_word_matching_filter)

                    elif filter_name.strip() == "item":
                        filters_list.append(item_filter)

                    elif filter_name.strip() == "value range":
                        filters_list.append(value_range_filter)

                    elif filter_name.strip() == "word in":
                        filters_list.append(word_in_filter)

                    elif filter_name.strip() == "location":
                        filters_list.append(location_filter)
            else:
                small_dictionary[test_csv.columns[j]] = test_csv.iloc[i, j].split(",")

    recommended_item_list = []
    for index in range(len(item_names)):
        recommended_item_list.append(RecommendedItem(Item("", item_names[index], {}), "", [""]))

    large_dictionary["recommended_items"] = [recommended_item_list]
    large_dictionary["hard_constraints"] = small_dictionary
    state = CommonStateManager({}, data=large_dictionary)
    test_data.append((state, filters_list, expected_indices))

rec_item1 = RecommendedItem(Item("CF33F8-E6oudUQ46HnavjQ", "Sonic Drive-In", {}), "", [""])
rec_item2 = RecommendedItem(Item("MUTTqe8uqyMdBl186RmNeA", "Tuna Bar", {}), "", [""])


class TestFilterApplier:

    @pytest.mark.parametrize("state_manager, filters, expected_indices", test_data)
    def test_apply_filter(self, state_manager: CommonStateManager, filters: list[Filter],
                          expected_indices: list[int]):
        """
        Test exact word matching filter.

        :param state_manager: state
        :param expected_indices: expected indices must be kept in the dataframe returned by the filter
        """
        filter_applier = FilterApplier(metadata_wrapper, filters)
        actual_indices = filter_applier.apply_filter(state_manager)
        assert actual_indices == expected_indices

    @pytest.mark.parametrize("current_item, expected_index",
                             [(rec_item1, [1]), (rec_item2, [7])])
    def test_filter_by_current_item(self, current_item: RecommendedItem,
                                    expected_index: list[int]):
        """
        Test exact word matching filter.

        :param current_item: current item
        :param expected_index: expected index must be kept in the dataframe returned by the filter
        """
        filter_applier = FilterApplier(metadata_wrapper, [])
        actual_index = filter_applier.filter_by_current_item(current_item)
        assert actual_index == expected_index
