from geopy.distance import geodesic
from geopy.distance import great_circle


class CheckLocation:
    """
    Class that checks whether the location is close enough to a restaurant using their latitude and longitude.

    :param default_max_distance_in_km: maximum allowable distance between the restaurant and the location
    :param distance_type: if "geodesic", geodesic distance will be used for distance calculation.
    Otherwise, great circle distance will be used.
    """
    _default_max_distance_in_km: float
    _distance_type: str

    def __init__(self, default_max_distance_in_km: float, distance_type: str) -> None:
        self._default_max_distance_in_km = default_max_distance_in_km
        self._distance_type = distance_type

    def is_restaurant_close_enough_to_loc(self, lat_lon_of_loc: list[tuple[float, float]],
                                          lat_lon_of_restaurant: tuple[float, float],
                                          max_distance_in_km: list[float]) -> bool:
        """
        Check whether the distance between the restaurant and the location is within max distance.
        
        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is
        longitude of the location
        :param lat_lon_of_restaurant: tuple where the first element is latitude and the second element is
        longitude of the restaurant
        :param max_distance_in_km: maximum allowable distance
        :return: true or false on whether the distance between the restaurant and the location is within max distance
        """
        for index in range(len(lat_lon_of_loc)):
            # get the distance between the location and the restaurant in km
            if self._distance_type == "geodesic":
                distance_btw_loc_and_restaurant_in_km \
                    = self._get_geodesic_distance(lat_lon_of_loc[index], lat_lon_of_restaurant)
            else:
                distance_btw_loc_and_restaurant_in_km \
                    = self._get_great_circle_distance(lat_lon_of_loc[index], lat_lon_of_restaurant)

            if max(self._default_max_distance_in_km, max_distance_in_km[index]) \
                    >= distance_btw_loc_and_restaurant_in_km:
                return True
        return False

    def decide_max_dist_in_km(self, northeast: tuple[float, float], southwest: tuple[float, float]) -> float:
        """
        Decide maximum allowable distance in km based on how specific the location was provided by the user
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
    
    def _get_geodesic_distance(self, lat_lon_of_loc: tuple[float, float], 
                               lat_lon_of_restaurant: tuple[float, float]) -> float:
        """
        Get geodisic distance between the location and the restaurant using their latitudes and longitudes.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param lat_lon_of_restaurant: tuple where the first element is latitude and the second element is longitude of the restaurant
        :return: geodesic distance between the location and the restaurant in km
        """
        return geodesic(lat_lon_of_loc, lat_lon_of_restaurant).km
    
    def _get_great_circle_distance(self, lat_lon_of_loc: tuple[float, float], lat_lon_of_restaurant: tuple[float, float]) -> float:
        """
        Get great circle distance between the location and the restaurant using their latitudes and longitudes.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param lat_lon_of_restaurant: tuple where the first element is latitude and the second element is longitude of the restaurant
        :return: great circle distance between the location and the restaurant in km
        """
        return great_circle(lat_lon_of_loc, lat_lon_of_restaurant).km