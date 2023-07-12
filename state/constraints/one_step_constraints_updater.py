import re

import yaml

from intelligence.llm_wrapper import LLMWrapper
from state.constraints.constraints_updater import ConstraintsUpdater
from state.constraints.constraint_merger import ConstraintMerger
from state.state_manager import StateManager
from jinja2 import Environment, FileSystemLoader


class OneStepConstraintsUpdater(ConstraintsUpdater):

    """
    Class that updates constraints based on user's input using single prompt.

    :param llm_wrapper: llm used to update constraints
    :param constraints_categories: list of all possible constraint categories and its details
    :param few_shots: details including the input and ouput of the fewshot examples of the prompt
    :param domain: domain of the recommendation
    """

    def __init__(self, llm_wrapper: LLMWrapper, constraints_categories: list[dict],
                 few_shots: list[dict], domain: str, user_defined_constraint_mergers: list[ConstraintMerger]):
        self._llm_wrapper = llm_wrapper
        self._constraints_categories = constraints_categories
        self._constraint_keys = [
            constraint_category['key'] for constraint_category in constraints_categories]
        self._cumulative_constraints_keys = [constraint_category['key'] for constraint_category in
                                             constraints_categories if constraint_category['is_cumulative']]
        self._key_to_default_value = {constraint_category["key"]: constraint_category["default_value"] for constraint_category in constraints_categories}
        
        self._user_defined_constraint_mergers = user_defined_constraint_mergers
        self._domain = domain

        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env = Environment(loader=FileSystemLoader(
            config['CONSTRAINTS_PROMPT_PATH']))
        self.template = env.get_template(
            config['ONE_STEP_CONSTRAINTS_UPDATER_PROMPT_FILENAME'])
        
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
            state_manager.get("updated_keys")[
                "hard_constraints"] = updated_hard_constraints_keys
        if updated_soft_constraints_keys is not None and updated_soft_constraints_keys != {}:
            state_manager.get("updated_keys")[
                "soft_constraints"] = updated_soft_constraints_keys

        if state_manager.get("hard_constraints") is not None and new_constraints.get("hard_constraints") is not None:
            self._merge_constraints(state_manager.get("hard_constraints"), new_constraints.get("hard_constraints"),
                                    state_manager.get('updated_keys').get('hard_constraints', {}))
        if new_constraints.get("hard_constraints") == {}:
            state_manager.update("hard_constraints", None)
        else:
            state_manager.update("hard_constraints",
                                 new_constraints.get("hard_constraints"))

        if state_manager.get("soft_constraints") is not None and new_constraints.get("soft_constraints") is not None:
            self._merge_constraints(state_manager.get("soft_constraints"), new_constraints.get("soft_constraints"),
                                    state_manager.get('updated_keys').get('soft_constraints', {}))
        if new_constraints.get("soft_constraints") == {}:
            state_manager.update("soft_constraints", None)
        else:
            state_manager.update("soft_constraints",
                                 new_constraints.get("soft_constraints"))

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
        
        #Update constraint to default value if applicable
        for key, default_val in self._key_to_default_value.items():
            if default_val != 'None' and state_manager.get('hard_constraints') and state_manager.get('hard_constraints').get(key) is None:
                # Update hard constraints 
                state_manager.get('hard_constraints')[key] = [default_val]
                #Update updated keys
                state_manager.get("updated_keys")["hard_constraints"][key] = True

    def _generate_prompt(self, state_manager: StateManager) -> str:
        """
        Generate prompt for updating constraints.

        :param state_manager: current state
        :return: prompt for updating constraints
        """
        conv_history = state_manager.get('conv_history')
        curr_user_input = conv_history[-1].get_content() if len(
            conv_history) >= 1 else ""
        prev_rec_response = conv_history[-2].get_content() if len(
            conv_history) >= 2 else ""
        prev_user_input = conv_history[-3].get_content() if len(
            conv_history) >= 3 else ""

        return self.template.render(user_input=curr_user_input, prev_rec_response=prev_rec_response,
                                    prev_user_input=prev_user_input,
                                    hard_constraints=state_manager.get(
                                        "hard_constraints"),
                                    soft_constraints=state_manager.get(
                                        "soft_constraints"),
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

    def _merge_constraints(self, old_constraints: dict, new_constraints: dict, updated_keys: dict) -> None:
        """
        Merge the given old_constraint to new_constraints.

        :param old_constraints: old hard or soft constraints
        :param new_constraints: new hard or soft constraints
        :param updated_keys: updated keys in this constraints
        """
        for constraint_merger in self._user_defined_constraint_mergers:
            if constraint_merger.get_constraint() in updated_keys and constraint_merger.get_constraint() in new_constraints and constraint_merger.get_constraint() in old_constraints:
                new_constraints[constraint_merger.get_constraint()] = constraint_merger.merge_constraint(
                    old_constraints.get(constraint_merger.get_constraint()),
                    new_constraints.get(constraint_merger.get_constraint())
                )
        
        for key in new_constraints:
            if key not in self._cumulative_constraints_keys and key in updated_keys and key in old_constraints:
                # remove all constraints in old_constraints from new_constraints
                for item in old_constraints[key]:
                    if item in new_constraints[key]:
                        new_constraints[key].remove(item)
