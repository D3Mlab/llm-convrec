from information_retriever.filter.value_range_filter import ValueRangeFilter
from state.common_state_manager import CommonStateManager
import pandas as pd
import pytest

metadata = pd.read_json("test/information_retriever/filter/15_clothing_metadata.json", orient='records', lines=True)

test_csv = pd.read_csv("test/information_retriever/filter/test_value_range_filter.csv", encoding ="ISO-8859-1")
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
            if test_csv.columns[j] == "expected_indices":
                expected_indices = [int(value) for value in test_csv.iloc[i, j].split(",")]
            elif test_csv.columns[j] == "price range":
                small_dictionary[test_csv.columns[j]] = test_csv.iloc[i, j].split(",")

    large_dictionary["hard_constraints"] = small_dictionary
    state = CommonStateManager({}, data=large_dictionary)
    test_data.append((state, expected_indices))


class TestValueRangeFilter:

    @pytest.mark.parametrize("state_manager, expected_indices", test_data)
    def test_value_range_filter(self, state_manager: CommonStateManager, expected_indices: list[int]):
        """
        Test exact word matching filter.

        :param state_manager: state
        :param expected_indices: expected indices must be kept in the dataframe returned by the filter
        """
        value_range_filter = ValueRangeFilter("price range", "price")
        filtered_metadata = value_range_filter.filter(state_manager, metadata)
        assert filtered_metadata.index.tolist() == expected_indices
