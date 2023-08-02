from geopy import Location, Nominatim
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper
import time


class NominatimWrapper(GeocoderWrapper):
    """
    Wrapper for Nominatim geocoder.
    """

    _geocoder_history: dict[str, Location]
    _max_attempts: int

    def __init__(self, max_attempts: int = 5, mandatory_address_key='road', location_bias=None):
        super().__init__()
        self._geocoder = Nominatim(user_agent='d3m-2023-convrec-demo')
        self._mandatory_address_key = mandatory_address_key
        self._geocoder_history = {}
        self._max_attempts = max_attempts
        self._location_bias = location_bias

    def geocode(self, query, **kwargs) -> Location:
        """
        Convert the given query to location object from geopy.

        :param query: query used to convert to location object (e.g. 'toronto, ontario')
        :param kwargs: other arguments
        :return: location object corresponding to the given query
        """

        if self._location_bias is not None and self._location_bias.lower() not in query.lower():
            query = f'{query}, {self._location_bias}'

        if query not in self._geocoder_history:
            attempts = 0
            while attempts < self._max_attempts:
                try:
                    self._geocoder_history[query] = self._geocoder.geocode(query, **{**kwargs, **{'addressdetails': True, 'timeout': None}})
                    break
                except Exception:
                    attempts += 1
                    time.sleep(attempts * 10)

                if attempts == self._max_attempts:
                    return None

        return self._geocoder_history[query]

    def is_location_specific(self, location: Location) -> bool:
        """
        Return whether location is specific enough.

        :param location: input location
        :return: whether location is specific enough.
        """
        if location is None:
            return False
        return self._mandatory_address_key in location.raw['address']

    def merge_location_query(self, new_loc_query: str, old_loc_query: str) -> str | None:
        """
        Merge given location queries to single query.
        If old location doesn't contain new location, return None.

        :param new_loc_query: new location query that should be part of old location
        :param old_loc_query: old location query that should contain new location
        :return: merged query or None if old location doesn't contain new location
        """
        merged_location = self.geocode(f'{new_loc_query}, {old_loc_query}')

        if merged_location is None or merged_location.raw['importance'] < 0.3:
            return None
        if merged_location == self.geocode(old_loc_query):
            return None
        else:
            return f'{new_loc_query}, {old_loc_query}'

    def get_boundary(self, location: Location) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get boundary points (northeast point and southwest point) of a location.
        :param location: input location
        :return: tuple where the first element represents northeast point and the second element represents southwest
        point. Each point is represented in a tuple where the first element is the latitude and the second element is
        the longitude.
        """
        boundingbox = location.raw.get('boundingbox')
        northeast_lat_lon = (float(boundingbox[1]), float(boundingbox[3]))
        southwest_lat_lon = (float(boundingbox[0]), float(boundingbox[2]))
        return northeast_lat_lon, southwest_lat_lon

    def get_lat_lon_of_loc(self, location: Location) -> tuple[float, float]:
        """
        Get the latitude and longitude of a location.
        :param location: input location
        :return: a tuple where the first element is the latitude and the second element is the longitude
        """
        return location.latitude, location.longitude
