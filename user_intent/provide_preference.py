from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper

from state.state_manager import StateManager
from state.constraints.constraints_updater import ConstraintsUpdater
from user_intent.user_intent import UserIntent
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from jinja2 import Environment, FileSystemLoader
import yaml


class ProvidePreference(UserIntent):
    """
    Class representing Provide Preference user intent.

    :param constraints_updater: object used to update constraints based on the user's input
    :param current_restaurants_extractor: object used to extract the restaurant that the user is referring to from the users input
    :param geocoder_wrapper: wrapper used to geocode location
    :param default_location: default location to use
    """

    _constraints_updater: ConstraintsUpdater
    _current_restaurants_extractor: CurrentItemsExtractor
    _geocoder_wrapper: GeocoderWrapper
    _default_location: str | None

    def __init__(self, constraints_updater: ConstraintsUpdater,
                 current_restaurants_extractor: CurrentItemsExtractor,
                 geocoder_wrapper: GeocoderWrapper, default_location=None):
        self._constraints_updater = constraints_updater
        self._current_restaurants_extractor = current_restaurants_extractor
        self._geocoder_wrapper = geocoder_wrapper
        self._default_location = default_location

        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(
            config['PROVIDE_PREFERENCE_PROMPT_FILENAME'])

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        return "Provide Preference"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "User provides background information for the restaurant search, provides specific preference for " \
               "the desired item, or improves over-constrained/under-constrained preferences"

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.
        Extract the constraints in the most recent user's input and add them to the field in the state.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """

        # Update current restaurant
        reccommended_restaurants = curr_state.get("recommended_items")

        if reccommended_restaurants is not None and reccommended_restaurants != []:
            curr_res = self._current_restaurants_extractor.extract(
                reccommended_restaurants, curr_state.get("conv_history"))

            # If current restaurant is [] then just keep it the same
            if curr_res != []:
                curr_state.update("curr_items", curr_res)

        # Update constraints
        self._constraints_updater.update_constraints(curr_state)


        if 'location' in curr_state.get("updated_keys").get("hard_constraints", {}):
            self._update_location_type(curr_state)

        return curr_state

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of provide preference

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """

        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self.template.render(user_input=user_input)

        return prompt

    def _update_location_type(self, curr_state: StateManager):
        """
        update the location type in the state to None, 'invalid', 'valid', or 'specific'.

        :param curr_state: current state of the conversation
        """
        hard_constraints = curr_state.get('hard_constraints')
        if hard_constraints is None or hard_constraints.get('location') is None:
            curr_state.update('location_type', None)
            return
        locations = hard_constraints.get('location')
        if len(locations) == 0:
            curr_state.update('location_type', None)
            return
        geocoded_latest_location = self._geocoder_wrapper.geocode(
            locations[-1])
        if geocoded_latest_location is None:
            curr_state.update('location_type', 'invalid')
            return
        if self._geocoder_wrapper.is_location_specific(geocoded_latest_location):
            curr_state.update('location_type', 'specific')
            return
        curr_state.update('location_type', 'valid')
