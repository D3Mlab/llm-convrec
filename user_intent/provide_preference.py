from state.state_manager import StateManager
from state.constraints.constraint_status import ConstraintStatus
from state.constraints.constraints_updater import ConstraintsUpdater
from user_intent.user_intent import UserIntent
from jinja2 import Environment, FileSystemLoader, Template
import threading
from utility.thread_utility import start_thread


class ProvidePreference(UserIntent):
    """
    Class representing Provide Preference user intent.

    :param constraints_updater: object used to update constraints based on the user's input
    :param constraint_statuses: list of status objects used to represent the status of constraints
    :param config: config of the system
    """
    _constraints_updater: ConstraintsUpdater
    _constraint_statuses: list[ConstraintStatus]
    _template: Template
    _enable_threading: bool

    def __init__(self, constraints_updater: ConstraintsUpdater,
                 constraint_statuses: list[ConstraintStatus],
                 config: dict):
        self._constraints_updater = constraints_updater
                
        self._constraint_statuses = constraint_statuses

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self._template = env.get_template(
            config['PROVIDE_PREFERENCE_PROMPT_FILENAME'])
        
        self._enable_threading = config['ENABLE_MULTITHREADING']

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

    def update_state(self, curr_state: StateManager):
        """
        Mutate to update the curr_state and return them.
        Extract the constraints in the most recent user's input and add them to the field in the state.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        if self._enable_threading:
            constr_thread = threading.Thread(
                target=self._constraints_updater.update_constraints, args=(curr_state,))

            start_thread([constr_thread])
        else:
            self._constraints_updater.update_constraints(curr_state)

        # Update constraint status
        if self._constraint_statuses is not None:
            for constraint in self._constraint_statuses:
                constraint.update_status(curr_state)

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of provide preference

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self._template.render(user_input=user_input)
        return prompt
