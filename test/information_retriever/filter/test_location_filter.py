import dotenv
from domain_specific.classes.restaurants.location_filter import LocationFilter
from state.common_state_manager import CommonStateManager
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
import pandas as pd
import pytest

dotenv.load_dotenv()

metadata = pd.read_json("test/information_retriever/filter/50_restaurants_metadata.json", orient='records', lines=True)

test_csv = pd.read_csv("test/information_retriever/filter/test_location_filter.csv", encoding ="ISO-8859-1")
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
            elif test_csv.columns[j] == "location":
                small_dictionary[test_csv.columns[j]] = test_csv.iloc[i, j].split(",")

    large_dictionary["hard_constraints"] = small_dictionary
    state = CommonStateManager({}, data=large_dictionary)
    test_data.append((state, expected_indices))


class TestLocationFilter:

    @pytest.mark.parametrize("state_manager, expected_indices", test_data)
    def test_location_filter(self,  state_manager: CommonStateManager, expected_indices: list[int]):
        """
        Test exact word matching filter.

        :param state_manager: state
        :param expected_indices: expected indices must be kept in the dataframe returned by the filter
        """
        location_filter = LocationFilter("location", ["latitude", "longitude"], 2, GoogleV3Wrapper())
        filtered_metadata = location_filter.filter(state_manager, metadata)
        assert filtered_metadata.index.tolist() == expected_indices
