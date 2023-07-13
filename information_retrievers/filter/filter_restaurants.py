import pandas as pd
import torch
from state.state_manager import StateManager
from information_retrievers.filter.check_location import CheckLocation
from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.data_holder import DataHolder
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper
import ast
from itertools import chain

class FilterRestaurants:
    """
    Class that filters restaurants by constraints.
    
    :param check_location: checks whether the location specified by the user is close enough to each restaurant
    :param check_cuisine_type: checks whether the cuisine type specified by the user matches each restaurant
    :param check_already_recommended_restaurant: checks whether each restaurant is already recommended
    :param data_holder: holds all data needed for filter
    :param filter_constraints: constraints used to do filtering
    """
    _check_location: CheckLocation
    _check_cuisine_type: CheckCuisineDishType
    _check_already_recommended_restaurant: CheckAlreadyRecommendedRestaurant
    _data_holder: DataHolder
    _filter_constraints: list[str]

    def __init__(self, geocoder_wrapper: GeocoderWrapper, check_location: CheckLocation, check_cuisine_type: CheckCuisineDishType,
                 check_already_recommended_restaurant: CheckAlreadyRecommendedRestaurant, data_holder: DataHolder,
                 filter_constraints: list[str]):
        self._geocoder_wrapper = geocoder_wrapper
        self._check_location = check_location
        self._check_cuisine_type = check_cuisine_type
        self._check_already_recommended_restaurant = check_already_recommended_restaurant
        self._data_holder = data_holder
        self._filter_constraints = filter_constraints

    def filter_by_constraints(self, state_manager: StateManager) -> torch.Tensor:
        """
        Filter restaurants based on constraints.
        
        :param state_manager: current state
        :return: a tuple of matrix of review embeddings where embeddings for restaurants that do not meet constraints
        are replaced with zero vectors
        """
        # get each constraint from state_manager
        loc_names = state_manager.get("hard_constraints").get("location")
        cuisine_type = state_manager.get("hard_constraints").get("cuisine type")
        dish_type = state_manager.get("hard_constraints").get("dish type")
        if cuisine_type is None:
            if dish_type is None:
                cuisine_dish_type = None
            else:
                cuisine_dish_type = list(map(lambda x: x.lower().strip(), dish_type))
        else:
            if dish_type is None:
                cuisine_dish_type = list(map(lambda x: x.lower().strip(), cuisine_type))
            else:
                cuisine_dish_type = list(map(lambda x: x.lower().strip(), cuisine_type + dish_type))
        already_recommended_restaurant = []
        recommended_restaurant = state_manager.get('recommended_items')
        if recommended_restaurant is not None:
            already_recommended_restaurant = list(chain.from_iterable(recommended_restaurant))

        filtered_embedding_matrix = self._data_holder.get_item_embedding_matrix().detach().clone()
        restaurants_meta_data = self._data_holder.get_item_metadata().copy()
        num_of_reviews_per_restaurant = self._data_holder.get_num_of_reviews_per_item().detach().clone()
        lat_lon_of_loc = []
        max_distance_in_km = []
        constraints = self._filter_constraints
        for loc_name in loc_names:
            if "location" in constraints:
                location = self._geocoder_wrapper.geocode(loc_name)
                if location is None:
                    constraints.remove("location")
                else:
                    lat_lon_of_loc.append(self._geocoder_wrapper.get_lat_lon_of_loc(location))
                    northeast, southwest = self._geocoder_wrapper.get_boundary(location)
                    max_distance_in_km.append(self._check_location.decide_max_dist_in_km(northeast, southwest))

        num_of_restaurants = restaurants_meta_data.shape[0]
        reviews_so_far = 0

        for index in range(num_of_restaurants):
            lat_lon_of_restaurant = (restaurants_meta_data['latitude'].loc[index], restaurants_meta_data['longitude'].loc[index])
            categories_restaurant = list(restaurants_meta_data['categories'].loc[index].split(","))
            categories_restaurant = list(map(lambda x: x.lower().strip(), categories_restaurant))
            restaurant_name = restaurants_meta_data['name'].loc[index].lower().strip().replace(" ", "")

            # if the restaurant does not meet constraints, replace its review embeddings with zero vectors
            if self._should_remove_restaurant(lat_lon_of_loc, lat_lon_of_restaurant,
                                              cuisine_dish_type, categories_restaurant,
                                              restaurant_name, already_recommended_restaurant, constraints,
                                              max_distance_in_km):

                column_index_end_of_removing = reviews_so_far + int(num_of_reviews_per_restaurant[index])
                filtered_embedding_matrix[reviews_so_far:column_index_end_of_removing, :].zero_()

            reviews_so_far += int(num_of_reviews_per_restaurant[index])

        return filtered_embedding_matrix

    def _should_remove_restaurant(self, lat_lon_of_loc: list[tuple[float, float]], lat_lon_of_restaurant: tuple[float, float],
                                  cuisine_dish_type: list[str], categories_restaurant: list[str],
                                  restaurant_name: str, already_recommended_restaurant: list[RecommendedItem],
                                  constraints: list[str], max_distance_in_km: list[float]):
        """
        Check whether the restaurant should be removed.
        
        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param lat_lon_of_restaurant: tuple where the first element is latitude and the second element is longitude of the restaurant
        :param cuisine_dish_type: cuisine type and dish type specified by the user
        :param categories_restaurant: categories of restaurant in metadata
        :param restaurant_name: restaurant name
        :param already_recommended_restaurant: already recommended restaurants
        :param constraints: constraints used for filtering (e.g. "location", "cuisine type")
        :param already_recommended_restaurant: restaurants that are already recommended
        :param max_distance_in_km: maximum allowable distance for location filter in km
        :return: true if the restaurant should be removed, otherwise false
        """
        does_not_meet_location_constraints = "location" in constraints and \
            not self._check_location.is_restaurant_close_enough_to_loc(
                lat_lon_of_loc, lat_lon_of_restaurant, max_distance_in_km)

        does_not_meet_cuisine_type = "cuisine dish type" in constraints and \
            not self._check_cuisine_type.does_cuisine_dish_type_match(
                cuisine_dish_type, categories_restaurant)

        is_already_recommended = "already recommended restaurant" in constraints and \
            not self._check_already_recommended_restaurant.is_not_recommended(
                restaurant_name, already_recommended_restaurant)

        return does_not_meet_location_constraints or does_not_meet_cuisine_type or is_already_recommended


    def filter_by_restaurant_name(self, restaurants_names: list[str]) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Filter restaurants based on restaurant's name.
        
        :param restaurants_names: the name of restaurants
        :return: a tuple of matrix of review embeddings for restaurants and number of reviews per restaurant
        and reviews, review embeddings, and business ids of restaurants 
        """
        restaurants_meta_data = self._data_holder.get_item_metadata()
        related_restaurants_meta_data = pd.DataFrame()
        for name in restaurants_names:
             related_restaurant_meta_data = restaurants_meta_data[restaurants_meta_data.name.apply(str.lower) == name.casefold()]
             related_restaurants_meta_data = pd.concat([related_restaurants_meta_data, related_restaurant_meta_data])

        business_ids = related_restaurants_meta_data["business_id"].unique()
        num_of_reviews_per_restaurant = []
        restaurants_review_embeddings = self._data_holder.get_item_reviews_embedding()
        restaurant_review_embeddings = pd.DataFrame()
        embedding_matrix = []

        for business_id in business_ids:
            restaurant_review_embedding = \
                     restaurants_review_embeddings[restaurants_review_embeddings.Business_ID == business_id]
            restaurant_review_embeddings = pd.concat([restaurant_review_embeddings, restaurant_review_embedding])
            num_of_reviews_per_restaurant.append(torch.tensor(restaurant_review_embedding.shape[0]))
            embedded_reviews = restaurant_review_embedding["Embedding"]

            for embedded_review_string in embedded_reviews:
                embedded_review_list = ast.literal_eval(embedded_review_string)
                embedded_review_tensor = torch.tensor(embedded_review_list)
                embedding_matrix.append(embedded_review_tensor)

        # reset the indicies
        restaurant_review_embeddings.reset_index(drop=True, inplace=True)

        # convert to torch tensor
        num_of_reviews_per_restaurant = torch.stack(num_of_reviews_per_restaurant)
        embedding_matrix = torch.stack(embedding_matrix)

        return embedding_matrix, num_of_reviews_per_restaurant, restaurant_review_embeddings