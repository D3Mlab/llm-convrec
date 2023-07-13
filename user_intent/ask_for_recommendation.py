from state.state_manager import StateManager
from user_intent.user_intent import UserIntent
from jinja2 import Environment, FileSystemLoader
import yaml


class AskForRecommendation(UserIntent):
    """
    Class representing Ask For Recommendation user intent.
    """

    def __init__(self, config):

        env = Environment(loader=FileSystemLoader(
            config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(
            config['ACCEPT_RECOMMENDATION_PROMPT_FILENAME'])

    def get_name(self) -> str:
        """
        Returns the name of this user intent.

        :return: name of this user intent
        """
        return "Ask for Recommendation"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "User asks for a recommendation"

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        return curr_state

    def get_prompt_for_classification(self, curr_state: StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of ask for recommendation

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self.template.render(user_input=user_input)

        return prompt
