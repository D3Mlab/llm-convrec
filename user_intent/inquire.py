from state.state_manager import StateManager
from user_intent.user_intent import UserIntent
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from jinja2 import Environment, FileSystemLoader
import yaml


class Inquire(UserIntent):
    """
    Class representing Inquire user intent.

    :param current_restaurants_extractor: object used to extract the restaurant that the user is referring to from the users input
    """

    _current_restaurants_extractor: CurrentItemsExtractor

    def __init__(self, current_restaurants_extractor: CurrentItemsExtractor):
        self._current_restaurants_extractor = current_restaurants_extractor
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        env = Environment(loader=FileSystemLoader(config['INTENT_PROMPTS_PATH']))
        self.template = env.get_template(config['INQUIRE_PROMPT_FILENAME'])

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

    def update_state(self, curr_state: StateManager) -> StateManager:
        """
        Mutate to update the curr_state and return them.

        :param curr_state: current state representing the conversation
        :return: new updated state
        """
        
        # Update current restaurant
        reccommended_restaurants = curr_state.get("recommended_items")

        if reccommended_restaurants is not None and reccommended_restaurants != []:
            curr_res = self._current_restaurants_extractor.extract(reccommended_restaurants, curr_state.get("conv_history"))
            
            # If current restaurant is [] then just keep it the same
            if curr_res != []:
                curr_state.update("curr_items", curr_res)
            
                
        return curr_state
    
    def get_prompt_for_classification(self, curr_state:StateManager) -> str:
        """
        Returns prompt for generating True/False representing how likely the user input matches with the user intent of inquire

        :param curr_state: current state representing the conversation
        :return: the prompt in string format
        """
        user_input = curr_state.get("conv_history")[-1].get_content()
        prompt = self.template.render(user_input=user_input)
        return prompt
