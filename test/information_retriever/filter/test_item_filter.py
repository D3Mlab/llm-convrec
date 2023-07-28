from information_retriever.filter.item_filter import ItemFilter
from state.common_state_manager import CommonStateManager
from information_retriever.item.item import Item
from information_retriever.item.recommended_item import RecommendedItem
import pandas as pd
import pytest

metadata = pd.read_json("test/information_retriever/filter/50_restaurants_metadata.json", orient='records', lines=True)

test_csv = pd.read_csv("test/information_retriever/filter/test_item_filter.csv", encoding ="ISO-8859-1")
num_rows = test_csv.shape[0]
num_columns = test_csv.shape[1]
test_data = []

for i in range(num_rows):
    large_dictionary = {}
    small_dictionary = {}
    for j in range(num_columns):
        if pd.isna(test_csv.iloc[i, j]):
            continue
        else:
            if test_csv.columns[j] == "key_in_state":
                constraint_keys = test_csv.iloc[i, j].strip()
            elif test_csv.columns[j] == "metadata_field":
                metadata_field = test_csv.iloc[i, j].strip()
            elif test_csv.columns[j] == "expected_removed_indices":
                expected_removed_indices = [int(value) for value in test_csv.iloc[i, j].split(",")]
            elif test_csv.columns[j] == "id_input":
                id_inputs = test_csv.iloc[i, j].split(",")
            elif test_csv.columns[j] == "name":
                names = test_csv.iloc[i, j].split(",")

    recommended_item_list = []
    for index in range(len(id_inputs)):
        recommended_item_list.append(RecommendedItem(Item(id_inputs[index], names[index], {}), "", [""]))

    large_dictionary["recommended_items"] = [recommended_item_list]
    state = CommonStateManager({}, data=large_dictionary)
    test_data.append((constraint_keys, metadata_field, state, expected_removed_indices))


class TestItemFilter:

    @pytest.mark.parametrize("constraint_key, metadata_field, state_manager, expected_removed_indices", test_data)
    def test_item_filter(self, constraint_key: str, metadata_field: str,
                         state_manager: CommonStateManager, expected_removed_indices: list[int]):
        """
        Test exact word matching filter.

        :param constraint_key: constraint keys of interest
        :param metadata_field: metadata field of interest
        :param state_manager: state
        :param expected_removed_indices: expected indices must not exist in the dataframe returned by the filter
        """
        item_filter = ItemFilter(constraint_key, metadata_field)
        filtered_metadata = item_filter.filter(state_manager, metadata)
        remained_indices = filtered_metadata.index.tolist()

        for value in expected_removed_indices:
            if value in remained_indices:
                assert False

        assert True
