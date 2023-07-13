import re
from typing import Any

from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from state.state_manager import StateManager
import yaml
from jinja2 import Environment, FileSystemLoader


class ConstraintsRemover:

    """
    Class responsible for removing constraints based on the user's input

    :param llm_wrapper: llm used to remove constraints
    :param default_keys: all possible keys for constraints
    """

    def __init__(self, llm_wrapper: LLMWrapper, default_keys: list):
        self._llm_wrapper = llm_wrapper
        self._default_keys = set(default_keys)
        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(
            config['CONSTRAINTS_PROMPT_PATH']))
        self.template = env.get_template(
            config['CONSTRAINTS_REMOVER_PROMPT_FILENAME'])

    def remove_ignored_constraints(self, state_manager: StateManager) -> dict:
        """
        Remove all constraints irrelevant, based on the most recent user's input.

        :param state_manager: current state
        """
        hard_constraints = state_manager.get("hard_constraints")
        soft_constraints = state_manager.get("soft_constraints")
        ignored_constraints = {}
        if hard_constraints is not None:
            prompt = self._generate_prompt(
                state_manager.get('conv_history'), hard_constraints)
            llm_response = self._llm_wrapper.make_request(prompt)
            new_constraints, ignored_constraints['hard_constraints'] = \
                self._update_constraints_from_llm_response(
                    llm_response, hard_constraints)
            state_manager.update("hard_constraints", new_constraints)
        if soft_constraints is not None:
            prompt = self._generate_prompt(
                state_manager.get('conv_history'), soft_constraints)
            llm_response = self._llm_wrapper.make_request(prompt)
            new_constraints, ignored_constraints['soft_constraints'] = \
                self._update_constraints_from_llm_response(
                    llm_response, soft_constraints)
            state_manager.update("soft_constraints", new_constraints)
        return ignored_constraints

    def _generate_prompt(self, conv_history: list[Message], constraints: dict) -> str:
        """
        Generate and return prompt for removing constraints.

        :param conv_history: history of the past conversation
        :param constraints: current constraints (hard or soft)
        :return: prompt for removing constraints
        """
        curr_user_input = conv_history[-1].get_content() if len(
            conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(
            conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(
            conv_history) >= 3 else ""

        return self.template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                    prev_user_input=prev_user_input,
                                    formatted_constraints=self._format_constraints(constraints))

    def _format_constraints(self, extracted_constraints: dict) -> str:
        """
        Format the given constraints as shown below:
         - <constraint key 1>: <constraint value 1>
         - <constraint key 2>: <constraint value 2>
         ...

        :param extracted_constraints: constraints extracted from the most recent user's input where each key represents
        the constraint category
        :return: formatted constraints
        """
        result = ""
        for key in extracted_constraints:
            values = ', '.join(
                f'"{value}"' for value in extracted_constraints[key])
            result += f" - {key}: {values}\n"
        return result.removesuffix('\n')

    def _update_constraints_from_llm_response(self, text: str, constraints: dict) -> tuple[dict, dict]:
        """
        Convert string formatted with
         - key1: item1, item2, ...
         - key2: item1, item2, ...
            ...
        to update the given constraints and return the updated constraints.
        :param text: text generated from llm formatted as shown above
        :param constraints: constraints extracted from the most recent user's input where each key represents
        the constraint category
        :return: updated constraints
        """
        ignored_constraints = {}
        for line in text.splitlines():
            line_arr = line.strip().removeprefix('-').strip().split(':')
            if len(line_arr) > 1:
                key = line_arr[0].strip()
                values_str = re.sub(r'\([^)]*\)', '', line_arr[1])
                values_str = values_str.strip().lower()
                if key in self._default_keys:
                    values_lst = [value.strip().removesuffix('"').removeprefix('"').removesuffix('.').strip() for value in
                                  re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', values_str)]
                    values_lst = list(filter(lambda a: a not in {'none', 'unspecified', 'unknown', 'not specified',
                                                                 'varied', 'undefined', 'general', 'n/a', '',
                                                                 'undecided', 'undetermined', 'none specified',
                                                                 'not provided', 'not specified/unknown'},
                                             values_lst))

                    if constraints.get(key) is None and values_lst != []:
                        constraints[key] = []
                    if values_lst:
                        for value in values_lst:
                            if value in constraints[key]:
                                if key not in ignored_constraints:
                                    ignored_constraints[key] = []
                                ignored_constraints[key].append(value)
                                constraints[key].remove(value)

        return constraints, ignored_constraints
