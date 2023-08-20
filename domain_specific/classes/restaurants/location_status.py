from state.state_manager import StateManager
from state.constraints.constraint_status import ConstraintStatus
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper


class LocationStatus(ConstraintStatus):
    """
    Class representing the status for location constraint

    :param _geocoder_wrapper: Wrapper for geocoding
    """
    _geocoder_wrapper: GeocoderWrapper
    
    def __init__(self, geocoder_wrapper: GeocoderWrapper):
        super().__init__("location")
        
        self._geocoder_wrapper = geocoder_wrapper

    def update_status(self, curr_state: StateManager) -> None:
        """
        Update the status of the constraint

        :param curr_state: current representation of the state
        :return: None
        """
        hard_constraints = curr_state.get('hard_constraints')
        if hard_constraints is None or hard_constraints.get('location') is None:
            self._curr_status = None
            return
        locations = hard_constraints.get('location')
        if len(locations) == 0:
            self._curr_status = None
            return
        geocoded_latest_location = self._geocoder_wrapper.geocode(
            locations[-1])
        if geocoded_latest_location is None:
            self._curr_status = "invalid"
        elif self._geocoder_wrapper.is_location_specific(geocoded_latest_location):
            self._curr_status = "specific"
        else:
            self._curr_status = "valid"

    def get_response_from_status(self) -> str | None:
        """
        Gets recommender response based off of constraints status
        Returns none if constraint is satisfied.

        :return: recommender response
        """
        if self._curr_status == "specific" or self._curr_status is None:
            return None
        elif self._curr_status == "invalid":
            return "I am sorry, I don't understand the given location. Could you give other location?"
        else:
            return "Could you provide more specific location?"

