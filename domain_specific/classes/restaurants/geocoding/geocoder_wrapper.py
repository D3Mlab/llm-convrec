from geopy import Location


class GeocoderWrapper:

    """
    Wrapper for geocoder.
    """

    def geocode(self, query: str, **kwargs) -> Location:
        """
        Convert the given query to location object from geopy.

        :param query: query used to convert to location object (e.g. 'toronto, ontario')
        :param kwargs: other arguments
        :return: location object corresponding to the given query
        """
        raise NotImplementedError()

    def is_location_specific(self, location: Location) -> bool:
        """
        Return whether location is specific enough.

        :param location: input location
        :return: whether location is specific enough.
        """
        raise NotImplementedError()

    def merge_location_query(self, new_loc_query: str, old_loc_query: str) -> str | None:
        """
        Merge given location queries to single query.
        If old location doesn't contain new location, return None.

        :param new_loc_query: new location query that should be part of old location
        :param old_loc_query: old location query that should contain new location
        :return: merged query or None if old location doesn't contain new location
        """
        raise NotImplementedError()

    def get_boundary(self, location: Location) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Get boundary points (northeast point and southwest point) of a location.
        :param location: input location
        :return: tuple where the first element represents northeast point and the second element represents southwest
        point. Each point is represented in a tuple where the first element is the latitude and the second element is
        the longitude.
        """
        raise NotImplementedError()

    def get_lat_lon_of_loc(self, location: Location) -> tuple[float, float]:
        """
        Get the latitude and longitude of a location.
        :param location: input location
        :return: a tuple where the first element is the latitude and the second element is the longitude
        """
        raise NotImplementedError()
