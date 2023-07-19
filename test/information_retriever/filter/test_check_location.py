from information_retrievers.filter.filter import CheckLocation
import pytest
import dotenv

dotenv.load_dotenv()

class TestCheckLocation:

    @pytest.mark.parametrize("lat_lon_of_loc, max_distance_in_km, lat_lon_of_restaurant, distance_type, "
                             "default_max_distance_in_km, expected_tf",
                             [([(43.651, -79.397)], [4.5], (43.680, -79.390), "geodesic", 2, True),
                              ([(42.983, -81.249)], [5], (42.990, -81.294), "great circle", 2, True),
                              ([(43.651, -79.397)], [2], (43.680, -79.390), "geodesic", 2, False),
                              ([(42.983, -81.249)], [3], (42.990, -81.294), "great circle", 2, False)])
    def test_is_restaurant_close_enough_to_loc(self, lat_lon_of_loc: tuple[float, float], max_distance_in_km: float, 
                                               lat_lon_of_restaurant: tuple[float, float], distance_type: str,
                                               default_max_distance_in_km: float, expected_tf: bool):
        """
        Test the function that checks whether the rsetaurant is close enough to the location using their latitude and longitude.

        :param lat_lon_of_loc: tuple where the first element is latitude and the second element is longitude of the location
        :param max_distance_in_km: maximum allowable distance between the restaurant and the location
        :param lat_lon_of_restaurant: tuple where the first element is latitude and the second element is longitude of the restaurant
        :param distance_type: "geodesic" or "great circle" to tell which type of distance should be used for distance calculation
        :param expected_tf: expected true or false
        """
        check_location = CheckLocation(default_max_distance_in_km, distance_type)
        assert check_location.is_restaurant_close_enough_to_loc(
            lat_lon_of_loc, lat_lon_of_restaurant, max_distance_in_km) == expected_tf
