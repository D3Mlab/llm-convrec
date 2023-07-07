import re

import yaml

from geocoding.geocoder_wrapper import GeocoderWrapper
from intelligence.llm_wrapper import LLMWrapper
from state.constraints.constraints_updater import ConstraintsUpdater
from state.state_manager import StateManager
from jinja2 import Environment, FileSystemLoader


class OneStepConstraintsUpdater(ConstraintsUpdater):

    """
    Class that updates constraints based on user's input using single prompt.

    :param llm_wrapper: llm used to update constraints
    :param geocoder_wrapper: wrapper used to geocode location
    :param constraints_categories: list of all possible constraint categories and its details
    :param few_shots: details including the input and ouput of the fewshot examples of the prompt
    :param domain: domain of the recommendation
    :param enable_location_merge: whether we merge location using geocoding
    """
    def __init__(self, llm_wrapper: LLMWrapper, geocoder_wrapper: GeocoderWrapper, constraints_categories: list[dict],
                 few_shots: list[dict], domain: str, enable_location_merge: bool = True):
        self._llm_wrapper = llm_wrapper
        self._geocoder_wrapper = geocoder_wrapper
        self._constraints_categories = constraints_categories
        self._constraint_keys = {constraint_category['key'] for constraint_category in constraints_categories}
        self._cumulative_constraints_keys = {constraint_category['key'] for constraint_category in
                                             constraints_categories if constraint_category['is_cumulative']}
        self._enable_location_merge = enable_location_merge
        self._domain = domain

        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(config['CONSTRAINTS_PROMPT_PATH']))
        self.template = env.get_template(config['ONE_STEP_CONSTRAINTS_UPDATER_PROMPT_FILENAME'])
        self._few_shots = few_shots

    def update_constraints(self, state_manager: StateManager) -> None:
        """
        Update the hard and soft constraints based on the most recent user's input.

        :param state_manager: current state
        """
        prompt = self._generate_prompt(state_manager)
        llm_response = self._llm_wrapper.make_request(prompt)
        new_constraints = self._format_llm_response(llm_response)

        updated_hard_constraints_keys = self._get_updated_keys_in_constraints(
                state_manager.get("hard_constraints"),
                new_constraints.get("hard_constraints")
            )

        updated_soft_constraints_keys = self._get_updated_keys_in_constraints(
                state_manager.get("soft_constraints"),
                new_constraints.get("soft_constraints")
            )
        if updated_hard_constraints_keys is not None and updated_hard_constraints_keys != {}:
            state_manager.get("updated_keys")["hard_constraints"] = updated_hard_constraints_keys
        if updated_soft_constraints_keys is not None and updated_hard_constraints_keys != {}:
            state_manager.get("updated_keys")["soft_constraints"] = updated_soft_constraints_keys

        if state_manager.get("hard_constraints") is not None and new_constraints.get("hard_constraints") is not None:
            self._merge_constraints(state_manager.get("hard_constraints"), new_constraints.get("hard_constraints"),
                                    state_manager.get('updated_keys').get('hard_constraints', {}))
        if new_constraints.get("hard_constraints") == {}:
            state_manager.update("hard_constraints", None)
        else:
            state_manager.update("hard_constraints", new_constraints.get("hard_constraints"))

        if state_manager.get("soft_constraints") is not None and new_constraints.get("soft_constraints") is not None:
            self._merge_constraints(state_manager.get("soft_constraints"), new_constraints.get("soft_constraints"),
                                    state_manager.get('updated_keys').get('soft_constraints', {}))
        if new_constraints.get("soft_constraints") == {}:
            state_manager.update("soft_constraints", None)
        else:
            state_manager.update("soft_constraints", new_constraints.get("soft_constraints"))

        if state_manager.get("soft_constraints") == {}:
            state_manager.update("soft_constraints", None)
        if state_manager.get("hard_constraints") == {}:
            state_manager.update("hard_constraints", None)
        if state_manager.get("soft_constraints") is not None:
            for key in set(state_manager.get("soft_constraints")):
                if not state_manager.get("soft_constraints")[key]:
                    state_manager.get('soft_constraints').pop(key)
        if state_manager.get("hard_constraints") is not None:
            for key in set(state_manager.get("hard_constraints")):
                if not state_manager.get("hard_constraints")[key]:
                    state_manager.get('hard_constraints').pop(key)

    def _generate_prompt(self, state_manager: StateManager) -> str:
        """
        Generate prompt for updating constraints.

        :param state_manager: current state
        :return: prompt for updating constraints
        """
        conv_history = state_manager.get('conv_history')
        curr_user_input = conv_history[-1].get_content() if len(conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(conv_history) >= 3 else ""

        return self.template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                    prev_user_input=prev_user_input,
                                    hard_constraints=state_manager.get("hard_constraints"),
                                    soft_constraints=state_manager.get("soft_constraints"),
                                    few_shots=self._few_shots,
                                    constraint_categories=self._constraints_categories,
                                    domain=self._domain)

    def _format_llm_response(self, llm_response: str) -> dict:
        """
        Format the response from the llm to dictionary.

        :param llm_response: response from the LLM
        :return: dictionary representation of the constraints
        """
        result = {"hard_constraints": {}, "soft_constraints": {}}
        key = None

        for line in llm_response.splitlines():
            if line.startswith("New Hard Constraints"):
                key = "hard_constraints"
                continue
            elif line.startswith("New Soft Constraints"):
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
                    if constraints_key in self._constraint_keys:
                        if result.get(constraints_key) is None:
                            result[key][constraints_key] = []
                        values_lst = [value.strip().removesuffix('"').removeprefix('"') for value in
                                      re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', values_str)]
                        values_lst = list(filter(lambda a: a not in {'none', 'unspecified', 'unknown', 'not specified',
                                                                     'varied', 'undefined', 'general', 'n/a', '',
                                                                     'undecided', 'undetermined', 'none specified',
                                                                     'not provided', 'not specified/unknown'},
                                                 values_lst))
                        result[key][constraints_key].extend(values_lst)
                        result[key][constraints_key] = list(
                            dict.fromkeys(result[key][constraints_key]))

        if 'hard_constraints' in result and len(result['hard_constraints']) == 0:
            result.pop('hard_constraints')
        if 'soft_constraints' in result and len(result['soft_constraints']) == 0:
            result.pop('soft_constraints')

        return result

    def _get_updated_keys_in_constraints(self, old_constraints, new_constraints):
        """
        Compute updated keys based on the given old and new constraints.

        :param old_constraints: old soft or hard constraints
        :param new_constraints: new soft or hard constraints
        :return: updated keys
        """
        if new_constraints is None:
            return None
        result = {}
        for new_key in new_constraints:
            if old_constraints is None or new_key not in old_constraints:
                result[new_key] = True
            else:
                for new_value in new_constraints[new_key]:
                    if new_value not in old_constraints[new_key]:
                        result[new_key] = True
                        break
        return result

    def _merged_location(self, old_locations: list[str], new_locations: list[str]) -> list[str]:
        """
        Return locations where new_locations are merged with old_locations using geocoding.
        New location is merged with most recently added location in old_locations if it can be merged.

        location merged in old_locations will be removed.

        :param old_locations: original locations
        :param new_locations: new locations that's added
        :return merged locations
        """
        merged_locations = []
        for new_location in new_locations:
            if new_location in old_locations:
                continue
            location_merged = False
            for i in range(len(old_locations) - 1, -1, -1):
                old_location = old_locations[i]
                if old_location in new_location:
                    old_locations.pop(i)
                    merged_locations.append(new_location)
                    location_merged = True
                    break
                else:
                    merged_location = self._geocoder_wrapper.merge_location_query(new_location, old_location)
                    if merged_location is not None:
                        old_locations.pop(i)
                        merged_locations.append(merged_location)
                        location_merged = True
                        break
            if not location_merged:
                merged_locations.append(new_location)
        if 'location' in self._cumulative_constraints_keys:
            return old_locations + merged_locations
        else:
            return merged_locations

    def _merge_constraints(self, old_constraints: dict, new_constraints: dict, updated_keys: dict) -> None:
        """
        Merge the given old_constraint to new_constraints.

        :param old_constraints: old hard or soft constraints
        :param new_constraints: new hard or soft constraints
        :param updated_keys: updated keys in this constraints
        """
        if self._enable_location_merge and 'location' in updated_keys and 'location' in new_constraints \
                and old_constraints.get('location') is not None:
            new_constraints['location'] = self._merged_location(
                old_constraints.get('location'),
                new_constraints.get('location')
            )

        for key in new_constraints:
            if key not in self._cumulative_constraints_keys and key in updated_keys and \
                    (key != 'location' or self._enable_location_merge) and key in old_constraints:
                # remove all constraints in old_constraints from new_constraints
                for item in old_constraints[key]:
                    if item in new_constraints[key]:
                        new_constraints[key].remove(item)

