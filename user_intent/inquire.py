from state.state_manager import StateManager
from user_intent.user_intent import UserIntent
from jinja2 import Environment, FileSystemLoader, Template


class Inquire(UserIntent):
    """
    Class representing Inquire user intent.

    :param few_shots: few shot examples used in the prompt
    :param domain: domain of the recommendation (e.g. "restaurants")
    :param config: config of the system
    """

    _few_shots: list[dict]
    _domain: str
    _template: Template

    def __init__(self, few_shots: list[dict], domain: str,
                 config: dict):
        env = Environment(loader=FileSystemLoader(config['INTENT_PROMPTS_PATH']))
        self._template = env.get_template(config['INQUIRE_PROMPT_FILENAME'])
        
        self._few_shots = few_shots
        self._domain = domain

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        return "Inquire"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "User requires additional information regarding the recommendation"

    def update_state(self, curr_state: StateManager):
        """
        Inquire does not need to update state

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        pass

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of
        inquire

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self._template.render(user_input=user_input, few_shots=self._few_shots, domain=self._domain)
        return prompt
