from state.constraints.constraint_merger import ConstraintMerger
from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper


class LocationConstraintMerger(ConstraintMerger):

    """
    Merge the new and old location constraints

    :param _geocoder_wrapper: Wrapper for geocoding
    """
    _geocoder_wrapper: GeocoderWrapper

    def __init__(self, geocoder_wrapper: GeocoderWrapper):
        super().__init__("location")
        self._geocoder_wrapper = geocoder_wrapper

    def merge_constraint(self, og_constraint_value: list[str], new_constraint_value: list[str]) -> list[str]:
        """
        Return locations where new_locations are merged with old_locations using geocoding.
        New location is merged with most recently added location in old_locations if it can be merged.

        location merged in old_locations will be removed.

        :param og_constraint_value: original locations
        :param new_constraint_value: new locations that's added
        :return merged locations
        """
        merged_locations = []
        for new_location in new_constraint_value:
            if new_location in og_constraint_value:
                continue
            location_merged = False
            for i in range(len(og_constraint_value) - 1, -1, -1):
                old_location = og_constraint_value[i]
                if old_location in new_location:
                    og_constraint_value.pop(i)
                    merged_locations.append(new_location)
                    location_merged = True
                    break
                else:
                    merged_location = self._geocoder_wrapper.merge_location_query(
                        new_location, old_location)
                    if merged_location is not None:
                        og_constraint_value.pop(i)
                        merged_locations.append(merged_location)
                        location_merged = True
                        break
            if not location_merged:
                merged_locations.append(new_location)

        return merged_locations
