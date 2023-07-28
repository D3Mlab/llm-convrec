import os
import dotenv
import pandas as pd
import pytest
import re

import yaml

from intelligence.gpt_wrapper import GPTWrapper
from state.common_state_manager import CommonStateManager
from state.constraints.one_step_constraints_updater import OneStepConstraintsUpdater
from state.message import Message
from domain_specific_config_loader import DomainSpecificConfigLoader
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper

dotenv.load_dotenv()


def load_dict(data_string):
    pattern = r'([^,]+)\s*=\s*\[([^\]]+)\]'
    matches = re.findall(pattern, data_string)
    data_dict = {key.strip(): [value.strip().removesuffix('"').removeprefix('"').removesuffix('.').strip().lower() for value in
                       re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lst)] for key, lst in matches}
    return data_dict


def parse_data(filepath):
    test_df = pd.read_csv(filepath, encoding='latin1')
    return [
        (
            [row[f'utterance {i}'] for i in range(1, 4) if f'utterance {i}' in row and isinstance(row[f'utterance {i}'], str)],
            load_dict(row["old hard constraints"]) if isinstance(row["old hard constraints"], str) else None,
            load_dict(row["old soft constraints"]) if isinstance(row["old soft constraints"], str) else None,
            load_dict(row["new hard constraints"]) if isinstance(row["new hard constraints"], str) else None,
            load_dict(row["new soft constraints"]) if isinstance(row["new soft constraints"], str) else None,
        )
        for row in test_df.to_dict("records")
    ]


# choose 'restaurant' or 'clothing' to configure which test to run
domain = 'restaurant'

test_data = parse_data(f'test/{domain}_constraints_updater_test.csv')
path_to_domain_configs = f'domain_specific/configs/{domain}_configs'


class TestConstraintsUpdater:

    @pytest.fixture(params=[
        (GPTWrapper(os.environ['OPENAI_API_KEY'], temperature=0)),
        (AlpacaLoraWrapper(os.environ['GRADIO_URL'], temperature=0))
    ])
    def updater(self, request):
        with open('system_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['PATH_TO_DOMAIN_CONFIGS'] = path_to_domain_configs
        domain_specific_config_loader = DomainSpecificConfigLoader(config)
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()
        yield OneStepConstraintsUpdater(request.param, constraints_categories,
                                        constraints_fewshots, domain_specific_config_loader.load_domain(), [],
                                        domain_specific_config_loader.system_config)

    @pytest.mark.parametrize('utterances,old_hard_constraints,old_soft_constraints,new_hard_constraints,new_soft_constraints', test_data)
    def test_single_turn_constraints_update_with_classification(self, updater, utterances, old_hard_constraints,
                                                                old_soft_constraints, new_hard_constraints,
                                                                new_soft_constraints) -> None:
        conv_history = []
        role = "user"
        for i in range(len(utterances)):
            conv_history.append(Message(role, utterances[i]))
            role = "recommender" if role == "user" else "user"

        state_manager = CommonStateManager(set())
        state_manager.update('conv_history', conv_history)
        state_manager.update('updated_keys', {})
        state_manager.update("hard_constraints", old_hard_constraints)
        state_manager.update("soft_constraints", old_soft_constraints)
        updater.update_constraints(state_manager)

        actual = {
            "hard_constraints": state_manager.get("hard_constraints"),
            "soft_constraints": state_manager.get("soft_constraints")
        }
        expected = {
            "hard_constraints": new_hard_constraints,
            "soft_constraints": new_soft_constraints
        }
        assert actual == expected

    @pytest.mark.parametrize('utterances,old_hard_constraints,old_soft_constraints,new_hard_constraints,new_soft_constraints', test_data)
    def test_single_turn_constraints_update_without_classification(self, updater, utterances, old_hard_constraints, old_soft_constraints,
                                                                    new_hard_constraints, new_soft_constraints) -> None:
        conv_history = []
        role = "user"
        for i in range(len(utterances)):
            conv_history.append(Message(role, utterances[i]))
            role = "recommender" if role == "user" else "user"

        state_manager = CommonStateManager(set())
        state_manager.update('conv_history', conv_history)
        state_manager.update('updated_keys', {})
        state_manager.update("hard_constraints", old_hard_constraints)
        state_manager.update("soft_constraints", old_soft_constraints)
        updater.update_constraints(state_manager)

        actual = self.merge_constraints(state_manager.get("hard_constraints"), state_manager.get("soft_constraints"))
        expected = self.merge_constraints(new_hard_constraints, new_soft_constraints)
        if actual == {}:
            actual = None
        if expected == {}:
            expected = None

        assert actual == expected

    @staticmethod
    def merge_constraints(hard_constraints, soft_constraints):
        if hard_constraints is None:
            return soft_constraints
        if soft_constraints is None:
            return hard_constraints
        result = {}
        for key in soft_constraints:
            if key not in hard_constraints:
                result[key] = soft_constraints[key]
            else:
                result[key] = soft_constraints[key] + hard_constraints[key]

        for key in hard_constraints:
            if key not in soft_constraints:
                result[key] = hard_constraints[key]

        return result
