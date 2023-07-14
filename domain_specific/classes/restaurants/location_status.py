from state.state_manager import StateManager
from state.status import Status
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper


class LocationStatus(Status):
    """
    Class representing the status for location constraint
    """
    _constraint: str
    _status_types: list[str]
    _state_key: str
    
    def __init__(self):
        super().__init__("location")
        
        self._geocoder_wrapper = GoogleV3Wrapper()

            
    def update_status(self, curr_state: StateManager):
        """
        update the location type in the state to None, 'invalid', 'valid'.

        :param curr_state: current state of the conversation
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
        if geocoded_latest_location is None or self._geocoder_wrapper.is_location_specific(geocoded_latest_location):
            self._curr_status = "invalid"
            return
        
        self._curr_status = "valid"