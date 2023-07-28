from state.state_manager import StateManager
from geopy.distance import geodesic
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper
from information_retriever.filter.filter import Filter
import pandas as pd


class LocationFilter(Filter):
    """
    Responsible to do filtering by checking whether the item is within max_distance
    from the location ( or one of the location) specified by the user

    :param constraint_key: constraint key of interest
    :param metadata_field: metadata field of interest
    :param default_max_distance_in_km: default max allowable distance in km
    """

    _constraint_key: str
    _metadata_field: list[str]
    _default_max_distance_in_km: float
    _geocoder_wrapper: GeocoderWrapper

    def __init__(self, constraint_key: str, metadata_field: list[str],
                 default_max_distance_in_km: float, geocoder_wrapper: GeocoderWrapper) -> None:
        self._constraint_key = constraint_key
        self._metadata_field = metadata_field
        self._default_max_distance_in_km = default_max_distance_in_km
        self._geocoder_wrapper = geocoder_wrapper

    def filter(self, state_manager: StateManager,
               metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Return a filtered version of metadata pandas dataframe.

        :param state_manager: current state
        :param metadata: items' metadata
        :return: filtered version of metadata pandas dataframe
        """
        location_names = state_manager.get('hard_constraints').get(self._constraint_key)
        if location_names is None or not location_names:
            return metadata

        lat_lon_of_locations, max_distances_in_km = self._get_lat_lon_and_max_distance(location_names)
        if not lat_lon_of_locations or not max_distances_in_km:
            return metadata

        metadata['is_close_enough'] = metadata.apply(
            self._is_item_close_enough_to_loc, args=(lat_lon_of_locations, max_distances_in_km), axis=1)
        filtered_metadata = metadata.loc[metadata['is_close_enough']]
        filtered_metadata = filtered_metadata.drop('is_close_enough', axis=1)

        return filtered_metadata

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

    def _is_item_close_enough_to_loc(self, row_of_df: pd.Series,
                                     lat_lon_of_loc: list[tuple[float, float]],
                                     max_distance_in_km: list[float]) -> bool:
        """
        Check whether the distance between the item and the location is within max distance.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is
        longitude of the location
        :param max_distance_in_km: maximum allowable distance
        :return: true or false on whether the distance between the item and the location is within max distance
        """
        lat_lon_of_item = (row_of_df[self._metadata_field[0]], row_of_df[self._metadata_field[1]])

        for index in range(len(lat_lon_of_loc)):
            distance_btw_loc_and_item_in_km \
                = self._get_geodesic_distance(lat_lon_of_loc[index], lat_lon_of_item)

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
