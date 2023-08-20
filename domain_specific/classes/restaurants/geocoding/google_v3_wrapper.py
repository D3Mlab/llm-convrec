from geopy import GoogleV3, Location
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper

import os
import dotenv
dotenv.load_dotenv()


class GoogleV3Wrapper(GeocoderWrapper):

    """
    Wrapper for GoogleV3 geocoder.

    :param mandatory_address_keys: key used to determine if location is specific enough
    """

    _geocoder: GoogleV3
    _mandatory_address_keys: dict[str]
    _geocoder_history: dict[str, Location]

    def __init__(self, mandatory_address_keys=None):
        super().__init__()
        if mandatory_address_keys is None:
            mandatory_address_keys = {'route', 'intersection'}
        self._geocoder = GoogleV3(api_key=os.environ['GOOGLE_API_KEY'])
        self._mandatory_address_keys = mandatory_address_keys
        self._geocoder_history = {}

    def geocode(self, query, **kwargs) -> Location:
        """
        Convert the given query to location object from geopy.

        :param query: query used to convert to location object (e.g. 'toronto, ontario')
        :param kwargs: other arguments
        :return: location object corresponding to the given query
        """
        if query not in self._geocoder_history:
            self._geocoder_history[query] = self._geocoder.geocode(query, **kwargs)
        return self._geocoder_history[query]

    def is_location_specific(self, location: Location) -> bool:
        """
        Return whether location is specific enough.

        :param location: input location
        :return: whether location is specific enough.
        """
        if location is None:
            return False
        return any(any(adr_type in self._mandatory_address_keys for adr_type in address['types'])
                   for address in location.raw['address_components'])

    def merge_location_query(self, new_loc_query: str, old_loc_query: str) -> str | None:
        """
        Merge given location queries to single query.
        If old location doesn't contain new location, return None.

        :param new_loc_query: new location query that should be part of old location
        :param old_loc_query: old location query that should contain new location
        :return: merged query or None if old location doesn't contain new location
        """
        merged_location = self._geocoder.geocode(f'{new_loc_query}, {old_loc_query}')
        if merged_location is None or merged_location.raw.get('partial_match'):
            return None
        if merged_location == self._geocoder.geocode(old_loc_query):
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
        viewport = location.raw.get('geometry').get('viewport')
        northeast = viewport.get('northeast')
        northeast_lat_lon = (northeast.get('lat'), northeast.get('lng'))
        southwest = viewport.get('southwest')
        southwest_lat_lon = (southwest.get('lat'), southwest.get('lng'))
        return northeast_lat_lon, southwest_lat_lon

    def get_lat_lon_of_loc(self, location: Location) -> tuple[float, float]:
        """
        Get the latitude and longitude of a location.
        :param location: input location
        :return: a tuple where the first element is the latitude and the second element is the longitude
        """
        return location.latitude, location.longitude
