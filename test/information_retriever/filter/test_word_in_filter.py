from information_retriever.filter.word_in_filter import WordInFilter
from state.common_state_manager import CommonStateManager
import pandas as pd
import pytest

metadata = pd.read_json("test/information_retriever/filter/50_restaurants_metadata.json", orient='records', lines=True)

test_csv = pd.read_csv("test/information_retriever/filter/test_word_in_filter.csv", encoding ="ISO-8859-1")
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
                constraint_keys = [key.strip() for key in test_csv.iloc[i, j].split(",")]
            elif test_csv.columns[j] == "metadata_field":
                metadata_field = test_csv.iloc[i, j].strip()
            elif test_csv.columns[j] == "expected_indices":
                expected_indices = [int(value) for value in test_csv.iloc[i, j].split(",")]
            else:
                small_dictionary[test_csv.columns[j]] = test_csv.iloc[i, j].split(",")

    large_dictionary["hard_constraints"] = small_dictionary
    state = CommonStateManager({}, data=large_dictionary)
    test_data.append((constraint_keys, metadata_field, state, expected_indices))


class TestWordInFilter:

    @pytest.mark.parametrize("constraint_keys, metadata_field, state_manager, expected_indices", test_data)
    def test_word_in_filter(self, constraint_keys: list[str], metadata_field: str,
                                        state_manager: CommonStateManager, expected_indices: list[int]):
        """
        Test exact word matching filter.

        :param constraint_keys: constraint keys of interest
        :param metadata_field: metadata field of interest
        :param state_manager: state
        :param expected_indices: expected indices must be kept in the dataframe returned by the filter
        """
        word_in_filter = WordInFilter(constraint_keys, metadata_field)
        filtered_metadata = word_in_filter.filter(state_manager, metadata)
        assert filtered_metadata.index.tolist() == expected_indices
