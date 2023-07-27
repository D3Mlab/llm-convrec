import os
import dotenv
import pandas as pd
import pytest
import re

from intelligence.gpt_wrapper import GPTWrapper
from state.common_state_manager import CommonStateManager
from state.constraints.constraints_classifier import ConstraintsClassifier
from state.constraints.constraints_remover import ConstraintsRemover
from state.constraints.key_value_pair_constraints_extractor import KeyValuePairConstraintsExtractor
from state.constraints.one_step_constraints_updater import OneStepConstraintsUpdater
from state.constraints.safe_constraints_remover import SafeConstraintsRemover
from state.constraints.three_steps_constraints_updater import ThreeStepsConstraintsUpdater
from state.message import Message
from domain_specific_config_loader import DomainSpecificConfigLoader


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
domain = 'clothing'

test_data = parse_data(f'test/{domain}_constraints_updater_test.csv')
path_to_domain_configs = f'domain_specific/configs/{domain}_configs'


class TestConstraintsUpdater:

    @pytest.fixture()
    def cumulative_constraints(self):
        yield {"dish type", "type of meal", "price range", "wait times", "atmosphere", "dietary restrictions", "others",}

    @pytest.fixture()
    def constraints(self):
        yield ["location", "cuisine type", "dish type", "type of meal", "price range", "wait times", "atmosphere",
               "dietary restrictions", "others"]

    @pytest.fixture()
    def constraint_descriptions(self):
        yield [
            'The desired location of the restaurants.',
            'The desired specific style of cooking or cuisine offered by the restaurants (e.g., "Italian", "Mexican", "Chinese"). This can be implicitly provided through dish type (e.g "italian" if dish type is "pizza")',
            'The desired menu item or dish in the restaurant that user shows interests.',
            'The desired category of food consumption associated with specific times of day (e.g., "breakfast", "lunch", "dinner").',
            'The preferred range of prices for the restaurants as specified by the user.',
            'The acceptable wait time for the user when dining at the restaurants.',
            'The preferred atmosphere or ambience of the restaurants.',
            'Any specific dietary limitations or restrictions the user may have.',
            'Any additional constraints or preferred features (e.g. "patio", "free wifi", "free parking", ...).',
        ]

    @pytest.fixture(params=[
        ('one_step_constraints_updater', GPTWrapper(os.environ['OPENAI_API_KEY'], temperature=0)),
    ])
    def updater(self, request, constraints, cumulative_constraints):
        domain_specific_config_loader = DomainSpecificConfigLoader()
        domain_specific_config_loader.system_config['PATH_TO_DOMAIN_CONFIGS'] = path_to_domain_configs
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()
        if request.param[0] == 'one_step_constraints_updater':
            yield OneStepConstraintsUpdater(request.param[1], constraints_categories,
                                            constraints_fewshots, domain_specific_config_loader.load_domain(), [],
                                            domain_specific_config_loader.system_config)
        else:
            constraints_extractor = KeyValuePairConstraintsExtractor(
                request.param[1], constraints)
            constraints_classifier = ConstraintsClassifier(request.param[1], constraints)

            if request.param[0] == 'three_steps_constraints_updater':
                constraints_remover = ConstraintsRemover(request.param[1], default_keys=constraints)
            else:
                constraints_remover = SafeConstraintsRemover(request.param[1], default_keys=constraints)
            yield ThreeStepsConstraintsUpdater(
                    constraints_extractor, constraints_classifier, request.param[2],
                    constraints_remover=constraints_remover,
                    cumulative_constraints=cumulative_constraints,
                    enable_location_merge=True)

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
