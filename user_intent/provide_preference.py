from domain_specific.classes.restaurants.geocoding.geocoder_wrapper import GeocoderWrapper
from domain_specific.classes.restaurants.location_status import Status

from state.state_manager import StateManager
from state.status import Status
from state.constraints.constraints_updater import ConstraintsUpdater
from user_intent.user_intent import UserIntent
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from jinja2 import Environment, FileSystemLoader
import threading
from utility.thread_utility import start_thread


class ProvidePreference(UserIntent):
    """
    Class representing Provide Preference user intent.

    :param constraints_updater: object used to update constraints based on the user's input
    :param current_items_extractor: object used to extract the items that the user is referring to from the users input
    :param constraint_statuses: list of status objects used to represent the status of constraints
    """

    _constraints_updater: ConstraintsUpdater
    _current_items_extractor: CurrentItemsExtractor
    _constraint_statuses: list[Status]

    def __init__(self, constraints_updater: ConstraintsUpdater,
                 current_items_extractor: CurrentItemsExtractor, constraint_statuses: list[Status], config: dict):
        self._constraints_updater = constraints_updater
        self._current_items_extractor = current_items_extractor
                
        self._constraint_statuses = constraint_statuses

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(
            config['PROVIDE_PREFERENCE_PROMPT_FILENAME'])
        
        if config['ENABLE_MULTITHREADING'] == True:
            self.enable_threading = True
        else:
            self.enable_threading = False

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
        return "User provides background information for the item search, provides specific preference for " \
               "the desired item, or improves over-constrained/under-constrained preferences"

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.
        Extract the constraints in the most recent user's input and add them to the field in the state.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        
        if (self.enable_threading):
            curr_item_thread = threading.Thread(
                target=self._update_curr_item, args=(curr_state,))

            constr_thread = threading.Thread(
                target=self._constraints_updater.update_constraints, args=(curr_state,))

            start_thread([curr_item_thread, constr_thread])
        else:
            self._update_curr_item(curr_state)
            self._constraints_updater.update_constraints(curr_state)
            
        
        # Update constraint status
        if self._constraint_statuses is not None:
            for constraint in self._constraint_statuses:
                constraint.update_status(curr_state)

        return curr_state

    def _update_curr_item(self, curr_state: StateManager):
        """
        Update the current item 

        :param curr_state: current state representing the conversation
        :return: None
        """
        reccommended_items = curr_state.get("recommended_items")

        if reccommended_items is not None and reccommended_items != []:
            curr_item = self._current_items_extractor.extract(
                reccommended_items, curr_state.get("conv_history"))

            # If current items are [] then just keep it the same
            if curr_item != []:
                curr_state.update("curr_items", curr_item)
        

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of provide preference

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """

        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self.template.render(user_input=user_input)

        return prompt
