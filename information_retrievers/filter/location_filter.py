from state.state_manager import StateManager
from geopy.distance import geodesic
from geopy.distance import great_circle
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper
from typing import Any

class LocationFilter:
    """
    Responsible to check whether the item match the constraint by checking
    whether the item is within max_distance from the location ( or one of the location)
    specified by the user

    :param constraint_key: constraint key of interest
    :param metadata_field: metadata field of interest
    :param default_max_distance_in_km: default max allowable distance in km
    :param distance_type: distance type (geodesic or great circle)
    :param geocoder_wrapper: geocoder
    """

    _constraint_key: str
    _metadata_field: list[str]
    _default_max_distance_in_km: float
    _distance_type: str
    _geocoder_wrapper: GeocoderWrapper

    def __init__(self, constraint_key: str, metadata_field: list[str],
                 default_max_distance_in_km: float, distance_type: str,
                 geocoder_wrapper: GeocoderWrapper) -> None:
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field
        self._default_max_distance_in_km = default_max_distance_in_km
        self._distance_type = distance_type
        self._geocoder_wrapper = geocoder_wrapper

    def filter(self, state_manager: StateManager,
               metadata_wrapper: Any, item_ids: list) -> list[str]:
        """
        Return true if the item is close enough to the location, false otherwise.
        If the value for the constraint key of interest is empty or none of the location is valid,
        it will return true.

        :param state_manager: current state
        :param metadata_wrapper: holds metadata
        :param item_ids: item ids remained after all other filters by checkers
        :return: true if the item is close enough to the location, false otherwise
        """
        location_names = state_manager.get('hard_constraints').get(self._constraint_key)
        if not location_names:
            return item_ids

        lat_lon_of_locations, max_distances_in_km = self._get_lat_lon_and_max_distance(location_names)
        if not lat_lon_of_locations or not max_distances_in_km:
            return item_ids

        item_id_to_keep = []
        for item_id in item_ids:
            item_metadata_dict = metadata_wrapper.get_item_dict_from_id(item_id)
            lat_lon_of_item = (item_metadata_dict[self._metadata_field[0]], item_metadata_dict[self._metadata_field[1]])

            if self._is_item_close_enough_to_loc(lat_lon_of_locations, lat_lon_of_item, max_distances_in_km):
                item_id_to_keep.append(item_id)

        return item_id_to_keep

    def _get_lat_lon_and_max_distance(self, location_names: list[str]) -> tuple[list, list]:
        """
        Return a list of latitude and longitude and a list of max distance in km from a list of locations.

        :param location_names: location names in state
        :return: a tuple where the first element is a list of latitude and longitude of locations and
        the second element is a list of max allowable distance in km
        """
        lat_lon_of_loc = []
        max_distance_in_km = []
        for location_name in location_names:
            location = self._geocoder_wrapper.geocode(location_name)
            if location is not None:
                lat_lon_of_loc.append(self._geocoder_wrapper.get_lat_lon_of_loc(location))
                northeast, southwest = self._geocoder_wrapper.get_boundary(location)
                max_distance_in_km.append(self._calculate_max_dist_in_km(northeast, southwest))

        return lat_lon_of_loc, max_distance_in_km

    def _is_item_close_enough_to_loc(self, lat_lon_of_loc: list[tuple[float, float]],
                                     lat_lon_of_item: tuple[float, float],
                                     max_distance_in_km: list[float]) -> bool:
        """
        Check whether the distance between the item and the location is within max distance.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is
        longitude of the location
        :param lat_lon_of_item: tuple where the first element is latitude and the second element is
        longitude of the item
        :param max_distance_in_km: maximum allowable distance
        :return: true or false on whether the distance between the item and the location is within max distance
        """
        for index in range(len(lat_lon_of_loc)):
            if self._distance_type == "geodesic":
                distance_btw_loc_and_item_in_km \
                    = self._get_geodesic_distance(lat_lon_of_loc[index], lat_lon_of_item)
            else:
                distance_btw_loc_and_item_in_km \
                    = self._get_great_circle_distance(lat_lon_of_loc[index], lat_lon_of_item)

            if max(self._default_max_distance_in_km, max_distance_in_km[index]) \
                    >= distance_btw_loc_and_item_in_km:
                return True
        return False

    def _calculate_max_dist_in_km(self, northeast: tuple[float, float],
                                  southwest: tuple[float, float]) -> float:
        """
        Calculate maximum allowable distance in km based on how specific the location was provided by the user
        :param northeast: northeast point where the first element is its latitude and the second element is
        its longitude
        :param southwest: northeast point where the first element is its latitude and the second element is
        its longitude
        :return: maximum allowable distance for location filter in km
        """
        if self._distance_type == "geodesic":
            diagonal_distance_in_km = self._get_geodesic_distance(northeast, southwest)
        else:
            diagonal_distance_in_km = self._get_geodesic_distance(northeast, southwest)
        return diagonal_distance_in_km / 2

    @staticmethod
    def _get_geodesic_distance(lat_lon_of_loc: tuple[float, float],
                               lat_lon_of_item: tuple[float, float]) -> float:
        """
        Get geodesic distance between the location and the item using their latitudes and longitudes.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param lat_lon_of_item: tuple where the first element is latitude and the second element is longitude of the item
        :return: geodesic distance between the location and the item in km
        """
        return geodesic(lat_lon_of_loc, lat_lon_of_item).km

    @staticmethod
    def _get_great_circle_distance(lat_lon_of_loc: tuple[float, float],
                                   lat_lon_of_item: tuple[float, float]) -> float:
        """
        Get great circle distance between the location and the item using their latitudes and longitudes.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param lat_lon_of_item: tuple where the first element is latitude and the second element is longitude of the item
        :return: great circle distance between the location and the item in km
        """
        return great_circle(lat_lon_of_loc, lat_lon_of_item).km