from information_retrievers.filter.check_location import CheckLocation
from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from information_retrievers.data_holder import DataHolder
from state.common_state_manager import CommonStateManager
import dotenv
import pandas as pd
import torch
import pytest
from itertools import chain

dotenv.load_dotenv()
geocoder_wrapper = GoogleV3Wrapper()
check_cuisine_type = CheckCuisineDishType()
check_already_recommended_restaurant = CheckAlreadyRecommendedRestaurant()
embedding_matrix: torch.Tensor = torch.load("information_retrievers/data/Top-50-Restaurants/matrix.pt")
num_of_reviews_per_restaurant: torch.Tensor = torch.load(
            "information_retrievers/data/Top-50-Restaurants/item.pt")
restaurants_review_embeddings: pd.DataFrame = pd.read_csv(
            "information_retrievers/data/Top-50-Restaurants/top_50_restaurants_review_embedding_sorted.csv")
restaurants_meta_data: pd.DataFrame = pd.read_csv(
            "information_retrievers/data/Top-50-Restaurants/top_50_restaurants_sorted.csv")
data_holder = DataHolder("information_retrievers/data/Top-50-Restaurants/top_50_restaurants_sorted.csv",
                         "information_retrievers/data/Top-50-Restaurants/top_50_restaurants_review_embedding_sorted.csv",
                         "information_retrievers/data/Top-50-Restaurants/matrix.pt",
                         "information_retrievers/data/Top-50-Restaurants/item.pt")
num_reviews_for_nashville = 6 + 10 + 162 + 601 + 161
num_reviews_for_largo = 106

dictionary_info1 = {"name": "Super Dog",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant1 = Item("4iRzR7OaS-QaSXuvYxEGKA", dictionary_info1)
rec_restaurant1 = RecommendedItem(restaurant1, "", [])

dictionary_info2 = {"name": "Sonic Drive-In",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant2 = Item("bBDDEgkFA1Otx9Lfe7BZUQ", dictionary_info2)
rec_restaurant2 = RecommendedItem(restaurant2, "", [])

dictionary_info3 = {"name": "Caviar & Bananas",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant3 = Item("lk9IwjZXqUMqqOhM774DtQ", dictionary_info3)
rec_restaurant3 = RecommendedItem(restaurant3, "", [])

dictionary_info4 = {"name": "The Green Pheasant",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant4 = Item("tMkwHmWFUEXrC9ZduonpTg", dictionary_info4)
rec_restaurant4 = RecommendedItem(restaurant4, "", [])

dictionary_info5 = {"name": "The Green Pheasant",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant5 = Item("tMkwHmWFUEX", dictionary_info5)
rec_restaurant5 = RecommendedItem(restaurant5, "", [])


class TestFilterRestaurants:

    @pytest.mark.parametrize("loc_name, default_max_distance_in_km, distance_type, cuisine_type, dish_type, "
                             "already_recommended_restaurant, constraints, expected_num_restaurants, "
                             "expected_num_reviews",
                             [(["Largo"], 5, "great circle", [], [], [], ["location"], 1, 106),
                              (["Largo"], 5, "great circle", ["italian"], [], [], ["location", "cuisine dish type"], 1, 106),
                              (["Largo"], 5, "great circle", ["japanese"], [], [], ["location", "cuisine dish type"], 0, 0),
                              (["Treasure Island, Florida"], 7, "great circle", ["french"], [""], [],
                               ["location", "cuisine dish type"], 1, 134),
                              (["Nashville"], 12, "geodesic", ["japanese"], [],
                               [[rec_restaurant1], [rec_restaurant2], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 1, 161),
                              (["Nashville"], 12, "geodesic", ["french"], [],
                               [[rec_restaurant1, rec_restaurant2], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 0, 0),
                              (["Philadelphia"], 12, "geodesic", [], ["sushi"], [],
                               ["location", "cuisine dish type"], 1, 250),
                              (["Hospital"], 12, "geodesic", ["Mediterranean"], [], [], ["cuisine dish type"], 1, 134),
                              (["Hospital"], 12, "geodesic", ["Mediterranean"], [], [],
                               ["location", "cuisine dish type"], 1, 134),
                              (["Hospital", "restaurant"], 12, "geodesic", ["Mediterranean"], [], [],
                               ["location", "cuisine dish type"], 1, 134),
                              (["Hospital", "restaurant"], 12, "geodesic", ["Mediterranean"], [], [],
                               ["cuisine dish type"], 1, 134),
                              (["Hospital", "Nashville"], 12, "geodesic", ["Mediterranean"], [], [],
                               ["cuisine dish type"], 1, 134),
                              (["Nashville", "Hospital"], 12, "geodesic", ["Mediterranean"], [], [],
                               ["cuisine dish type"], 1, 134),
                              (["Philadelphia", "Nashville"], 12, "geodesic", ["japanese"], [], [],
                               ["location", "cuisine dish type"], 2, 411),
                              (["Nashville"], 12, "geodesic", ["japanese"], [],
                               [[rec_restaurant1], [rec_restaurant2], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 1, 161),
                              (["Nashville"], 12, "geodesic", ["french"], [],
                               [[rec_restaurant1, rec_restaurant2], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 0, 0),
                              (["Nashville"], 12, "geodesic", ["japanese"], [],
                               [[rec_restaurant1], [rec_restaurant2], [rec_restaurant4]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 0, 0),
                              (["Nashville"], 12, "geodesic", ["japanese"], [],
                               [[rec_restaurant1], [rec_restaurant5], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 0, 0),
                              (["Nashville"], 12, "geodesic", ["japanese"], [],
                               [[rec_restaurant4], [rec_restaurant5], [rec_restaurant3]],
                               ["location", "cuisine dish type", "already recommended restaurant"], 0, 0)
                              ])
    def test_filter_by_constraints(self, loc_name: str, default_max_distance_in_km: float, distance_type: str,
                                   cuisine_type: list[str], dish_type: list[str],
                                   already_recommended_restaurant: list[RecommendedItem],
                                   constraints: list[str], expected_num_restaurants: int, expected_num_reviews: int):
        """
        Test the function that filters restaurants based on constraints.
        
        :param loc_name: the name of the location
        :param default_max_distance_in_km: maximum allowable distance between the restaurant and the location
        :param distance_type: "geodesic" or "great circle" to tell which type of distance should be used for distance calculation
        :param cuisine_type: cuisine type specified by the user
        :param dish_type: dish type specified by the user
        :param constraints: constraints used for filtering (e.g. "location", "cuisine type")
        :param already_recommended_restaurant: restaurants that are already recommended
        :param expected_num_restaurants: number of restaurants that is expceted after filtering
        :param expected_num_reviews: number of reviews that is expected after filtering
        """
        check_location = CheckLocation(default_max_distance_in_km, distance_type)
        filter_restaurants = FilterRestaurants(geocoder_wrapper, check_location, check_cuisine_type,
                                               check_already_recommended_restaurant,
                                               data_holder, constraints)
        state_manager = self._create_state(loc_name, cuisine_type, dish_type, already_recommended_restaurant)
        filtered_embedding_matrix = \
            filter_restaurants.filter_by_constraints(state_manager)
        non_zero_rows = torch.any(filtered_embedding_matrix != 0, dim=1)
        assert filtered_embedding_matrix.size()[1] == 768 and non_zero_rows.sum().item() == expected_num_reviews

    @pytest.mark.parametrize("restaurants_names, expected_num_restaurants, expected_num_reviews",
                             [(["Cheeseburger In Paradise"], 1, 20),
                              (["Crafty Crab", "Helena Avenue Bakery"], 2, 14 + 401)])
    def test_filter_by_restaurant_name(self, restaurants_names: list[str],
                                       expected_num_restaurants: int, expected_num_reviews: int):
        """
        Test the function that filters restaurants based on restaurant's name.
        
        :param restaurants_names: the name of restaurants
        :param expected_num_restaurants: number of restaurants that is expceted after filtering
        :param expected_num_reviews: number of reviews that is expected after filtering
        """
        check_location = CheckLocation(0, "")
        filter_restaurants = FilterRestaurants(geocoder_wrapper, check_location, check_cuisine_type,
                                               check_already_recommended_restaurant,
                                               data_holder, [""])
        filtered_embedding_matrix, filtered_num_of_reviews_per_restaurant, filtered_restaurants_review_embeddings = \
            filter_restaurants.filter_by_restaurant_name(restaurants_names)
        assert filtered_embedding_matrix.size()[0] == expected_num_reviews \
               and filtered_embedding_matrix.size()[1] == 768 \
               and filtered_num_of_reviews_per_restaurant.size()[0] == expected_num_restaurants \
               and filtered_restaurants_review_embeddings.shape[0] == expected_num_reviews \
               and filtered_restaurants_review_embeddings.shape[1] == 3

    def _create_state(self, loc_name: list[str], cuisine_type: list[str], dish_type: list[str],
                      already_recommended_restaurant: list[RecommendedItem]) -> CommonStateManager:
        hard_constraint_dict = {}
        if loc_name:
            hard_constraint_dict["location"] = loc_name
        if cuisine_type:
            hard_constraint_dict["cuisine type"] = cuisine_type
        if dish_type:
            hard_constraint_dict["dish type"] = dish_type
        state_dict = {}
        state_dict["hard_constraints"] = hard_constraint_dict
        state_dict["recommended_items"] = already_recommended_restaurant
        state_manager = CommonStateManager({}, data=state_dict)
        return state_manager
