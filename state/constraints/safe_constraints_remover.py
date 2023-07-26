import yaml

from intelligence.llm_wrapper import LLMWrapper
from state.constraints.constraints_remover import ConstraintsRemover
from state.message import Message
from state.state_manager import StateManager
from jinja2 import Environment, FileSystemLoader


class SafeConstraintsRemover(ConstraintsRemover):
    """
    Class responsible for removing constraints based on the user's input.
    It also checks whether we should remove any constraints before to make it safer.

    :param llm_wrapper: llm used to remove constraints
    :param default_keys: all possible keys for constraints
    """

    def __init__(self, llm_wrapper: LLMWrapper, constraints_categories: list, config: dict):
        default_keys = [constraints_category['key'] for constraints_category in constraints_categories]
        
        super().__init__(llm_wrapper, default_keys)
        
        env = Environment(loader=FileSystemLoader(
            config['CONSTRAINTS_PROMPT_PATH']))
        self.removal_check_template = env.get_template(
            config['CONSTRAINTS_REMOVE_CHECK_PROMPT_FILENAME'])

    def remove_ignored_constraints(self, state_manager: StateManager) -> dict:
        """
         Remove all constraints irrelevant, based on the most recent user's input.
         Check whether we should remove anything before removing using prompt.

         :param state_manager: current state
         """
        prompt = self._generate_check_prompt(state_manager.get('conv_history'))
        llm_res = self._llm_wrapper.make_request(prompt)
        if llm_res.strip().lower().removesuffix('"').removesuffix('”').startswith("yes"):
            return super(SafeConstraintsRemover, self).remove_ignored_constraints(state_manager)
        return {}

    def _generate_check_prompt(self, conv_history: list[Message]) -> str:
        """
        Generate prompt used for checking whether we should remove any constraints based on the user's current input.

        :param conv_history: history of conversation
        :return: prompt used for checking whether we should remove any constraints based on the user's current input
        """
        curr_user_input = conv_history[-1].get_content() if len(
            conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(
            conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(
            conv_history) >= 3 else ""

        return self.removal_check_template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                                  prev_user_input=prev_user_input)
