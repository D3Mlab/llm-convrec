import re

import yaml

from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from state.constraints.constraints_extractor import ConstraintsExtractor
from jinja2 import Environment, FileSystemLoader


class KeyValuePairConstraintsExtractor(ConstraintsExtractor):
    _llm_wrapper: LLMWrapper

    def __init__(self, llm_wrapper: LLMWrapper, default_keys: list[str]) -> None:
        super().__init__(default_keys)
        self._llm_wrapper = llm_wrapper
        self._default_keys = default_keys
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(config['CONSTRAINTS_PROMPT_PATH']))
        self.template = env.get_template(config['CONSTRAINTS_EXTRACTOR_PROMPT_FILENAME'])

    def extract(self, conv_history: list[Message]) -> dict:
        """
        Extract the constraints from the most recent user's input, and return them.

        :param conv_history: current conversation history
        :return: constraints that was updated in this function.
        """
        prompt = self._generate_constraints_update_prompt(conv_history)
        llm_response = self._llm_wrapper.make_request(prompt)

        return self._update_constraints_from_llm_response(llm_response)

    def _update_constraints_from_llm_response(self, text: str) -> dict:
        """
        Convert string formatted with
            key1: item1, item2, ...
            key2: item1, item2, ...
            ...
        to update the given constraints and return the updated constraints.
        :param text: text generated from llm formatted as shown above
        :return: updated constraints
        """
        result = {}
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

                    if result.get(key) is None and values_lst != []:
                        result[key] = []
                    if values_lst:
                        result[key].extend(values_lst)

        return result

    def _generate_constraints_update_prompt(self, conv_history: list[Message]) -> str:
        """
        Generate and return prompt for extracting constraints from the most recent user's input in the
        conversation history.
        :param conv_history: current conversation history
        :return: prompt for extracting constraints from the most recent user's input in the conversation history.
        """
        curr_user_input = conv_history[-1].get_content() if len(conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(conv_history) >= 3 else ""

        return self.template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                    prev_user_input=prev_user_input)
