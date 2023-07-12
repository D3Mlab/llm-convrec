import re
from typing import Any

import yaml

from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from jinja2 import Environment, FileSystemLoader


class ConstraintsClassifier:

    """
    Class responsible for classifying the given constraint to hard / soft constraints.

    :param llm_wrapper: Wrapper for llm used to classify constraints
    :param default_keys: all possible keys for constraints
    """

    _llm_wrapper: LLMWrapper
    _default_keys: set[str]

    def __init__(self, llm_wrapper: LLMWrapper, default_keys: list[str] = None):
        self._llm_wrapper = llm_wrapper
        self._default_keys = set(default_keys)
        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(
            config['CONSTRAINTS_PROMPT_PATH']))
        self.template = env.get_template(
            config['CONSTRAINTS_CLASSIFIER_PROMPT_FILENAME'])

    def classify(self, conv_history: list[Message], extracted_constraints: dict) -> dict:
        """
        Classify the given extracted constraints to hard / soft constraints.

        :param conv_history: list of past messages
        :param extracted_constraints: constraints extracted from the most recent user's input where each key represents
        the constraint category
        :return: dictionary where each constraint in extracted_constraints are moved to either
        "hard_constraints" or "soft_constraints"
        """
        prompt = self._generate_prompt(conv_history, extracted_constraints)
        llm_response = self._llm_wrapper.make_request(prompt)
        return self._format_llm_response(llm_response, extracted_constraints)

    def _generate_prompt(self, conv_history: list[Message], extracted_constraints: dict[str, Any]):
        """
        Generate prompt for classifying constraints.

        :param conv_history: list of past messages
        :param extracted_constraints: constraints extracted from the most recent user's input where each key represents
        the constraint category
        :return: prompt for classifying constraints.
        """
        curr_user_input = conv_history[-1].get_content() if len(
            conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(
            conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(
            conv_history) >= 3 else ""

        return self.template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                    prev_user_input=prev_user_input,
                                    formatted_constraints=self._format_constraints(extracted_constraints))

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

    def _format_llm_response(self, llm_response: str, extracted_constraints: dict) -> dict:
        """
        Format the response from the llm to dictionary.

        :param llm_response: response from the LLM
        :return: dictionary representation of the constraints
        """
        result = {"hard_constraints": {}, "soft_constraints": {}}
        key = None

        for line in llm_response.splitlines():
            if line.startswith("Hard Constraints"):
                key = "hard_constraints"
                continue
            elif line.startswith("Soft Preferences"):
                key = "soft_constraints"
                continue
            elif line.startswith("None"):
                key = None
                continue

            if key is not None:
                line_arr = line.strip().removeprefix('-').strip().split(':')
                if len(line_arr) > 1:
                    constraints_key = line_arr[0].strip()
                    values_str = re.sub(r'\([^)]*\)', '', line_arr[1])
                    values_str = values_str.strip().lower()
                    if constraints_key in self._default_keys and constraints_key in extracted_constraints:
                        if result.get(constraints_key) is None:
                            result[key][constraints_key] = []
                        values_lst = [value.strip().removesuffix('"').removeprefix('"') for value in
                                      re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', values_str)]
                        result[key][constraints_key].extend([value for value in values_lst if value in
                                                             extracted_constraints[constraints_key]])
                        result[key][constraints_key] = list(
                            dict.fromkeys(result[key][constraints_key]))

        for key in extracted_constraints:
            diff = [x for x in extracted_constraints[key] if x not in result.get('soft_constraints', {}).get(key, [])
                    and x not in result.get('hard_constraints', {}).get(key, [])]
            if diff:
                if key not in result['soft_constraints']:
                    result['soft_constraints'][key] = []
                result['soft_constraints'][key].extend(diff)

        if 'hard_constraints' in result and len(result['hard_constraints']) == 0:
            result.pop('hard_constraints')
        if 'soft_constraints' in result and len(result['soft_constraints']) == 0:
            result.pop('soft_constraints')

        return result
