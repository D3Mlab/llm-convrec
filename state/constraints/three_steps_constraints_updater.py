from geocoding.geocoder_wrapper import GeocoderWrapper
from state.state_manager import StateManager
from state.constraints.constraints_classifier import ConstraintsClassifier
from state.constraints.constraints_remover import ConstraintsRemover
from state.constraints.constraints_updater import ConstraintsUpdater
from state.constraints.constraints_extractor import ConstraintsExtractor


class ThreeStepsConstraintsUpdater(ConstraintsUpdater):

    """
    Class to update constraints using three prompts.

    :param constraints_extractor: object used to extract constraints from the user's input
    :param constraints_classifier: object used to classify if a constraint is a hard or soft constraint
    :param constraints_remover: object used to remove constraints based on the user's input
    :param geocoder_wrapper: wrapper used to geocode location
    :param cumulative_constraints: constraints that is cumulative
    :param enable_location_merge: whether we merge location using geocoding
    """

    _constraints_extractor: ConstraintsExtractor
    _constraints_classifier: ConstraintsClassifier
    _geocoder_wrapper: GeocoderWrapper
    _cumulative_constraints: set[str]

    def __init__(self, constraints_extractor: ConstraintsExtractor,
                 constraints_classifier: ConstraintsClassifier,
                 geocoder_wrapper: GeocoderWrapper,
                 constraints_remover: ConstraintsRemover | None = None,
                 cumulative_constraints: set = None,
                 enable_location_merge: bool = True):
        self._constraints_extractor = constraints_extractor
        self._constraints_classifier = constraints_classifier
        self._constraints_remover = constraints_remover
        self._geocoder_wrapper = geocoder_wrapper
        if cumulative_constraints is None:
            cumulative_constraints = set()
        self._cumulative_constraints = cumulative_constraints
        self._enable_location_merge = enable_location_merge

    def update_constraints(self, state_manager: StateManager) -> None:
        """
        Update the hard and soft constraints based on the most recent user's input.

        :param state_manager: current state
        """
        if self._constraints_remover is not None:
            ignored_constraints = self._constraints_remover.remove_ignored_constraints(state_manager)
        else:
            ignored_constraints = {}

        extracted_constraints = self._constraints_extractor.extract(state_manager.get("conv_history"))
        if extracted_constraints is not None and extracted_constraints != {}:
            classified_constraints = self._constraints_classifier.classify(
                state_manager.get('conv_history'),
                extracted_constraints
            )
            if "hard_constraints" in classified_constraints:
                if state_manager.get("hard_constraints") is None:
                    state_manager.update("hard_constraints", {})
                state_manager.get("updated_keys")["hard_constraints"] = \
                    self._merge_constraints(state_manager.get("hard_constraints"),
                                            classified_constraints["hard_constraints"],
                                            ignored_constraints.get('hard_constraints', {}))

            if "soft_constraints" in classified_constraints:
                if state_manager.get("soft_constraints") is None:
                    state_manager.update("soft_constraints", {})
                state_manager.get("updated_keys")["soft_constraints"] = \
                    self._merge_constraints(state_manager.get("soft_constraints"),
                                            classified_constraints["soft_constraints"],
                                            ignored_constraints.get('soft_constraints', {}))

        if state_manager.get("soft_constraints") == {}:
            state_manager.update("soft_constraints", None)
        if state_manager.get("hard_constraints") == {}:
            state_manager.update("hard_constraints", None)
        if state_manager.get("soft_constraints") is not None:
            for key in set(state_manager.get("soft_constraints")):
                if not state_manager.get("soft_constraints")[key]:
                    state_manager.get('soft_constraints').pop(key)
        if state_manager.get("hard_constraints") is not None:
            for key in set(state_manager.get("hard_constraints")):
                if not state_manager.get("hard_constraints")[key]:
                    state_manager.get('hard_constraints').pop(key)

    def _merge_location(self, old_locations: list[str], new_locations: list[str], ignored_locations: list[str]) -> list[str]:
        """
        Merge new location into old_locations

        :param old_locations: original locations
        :param new_locations: new locations that's added
        :param ignored_locations: all ignored locations
        :return: merged locations
        """
        result = []
        for new_location in new_locations:
            is_location_merged = False
            for i in range(len(old_locations) - 1, -1, -1):
                if old_locations[i] == new_location:
                    result.append(new_location)
                    old_locations.pop(i)
                    is_location_merged = True
                    break
                merged_location = self._geocoder_wrapper.merge_location_query(new_location, old_locations[i])
                if merged_location is not None:
                    result.append(merged_location)
                    old_locations.pop(i)
                    is_location_merged = True
                    break
            if not is_location_merged:
                for i in range(len(ignored_locations) - 1, -1, -1):
                    if ignored_locations[i] == new_location:
                        result.append(new_location)
                        is_location_merged = True
                        break
                    merged_location = self._geocoder_wrapper.merge_location_query(new_location, ignored_locations[i])
                    if merged_location is not None:
                        result.append(merged_location)
                        is_location_merged = True
                        break
                if not is_location_merged:
                    result.append(new_location)

        if 'location' in self._cumulative_constraints:
            return old_locations + result
        else:
            return result

    def _merge_constraints(self, curr_constraints: dict, classified_constraints: dict, ignored_constraints: dict) -> set[str]:
        """
        Merge the given classified constraints to current constraints.

        :param curr_constraints: hard or soft current constraints
        :param classified_constraints: hard or soft constraints that was extracted and classified from most recent
        user's input
        :param ignored_constraints: all the ignored soft or hard constraints
        :return: all the keys modified
        """
        modified_keys = set()
        for key in classified_constraints:
            modified_keys.add(key)
            if key not in curr_constraints or key not in self._cumulative_constraints and key != 'location':
                curr_constraints[key] = []
            if self._enable_location_merge and key == 'location':
                curr_constraints['location'] = self._merge_location(curr_constraints.get('location', []),
                                                                    classified_constraints['location'],
                                                                    ignored_constraints.get('location', []))
            else:
                curr_constraints[key].extend(classified_constraints[key])
                curr_constraints[key] = list(dict.fromkeys(curr_constraints[key]))
        return modified_keys


