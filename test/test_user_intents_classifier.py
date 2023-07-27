import pytest
import pandas as pd
from user_intent.classifiers.multilabel_user_intents_classifier import MultilabelUserIntentsClassifier
from user_intent.ask_for_recommendation import AskForRecommendation
from user_intent.inquire import Inquire
from user_intent.accept_recommendation import AcceptRecommendation
from user_intent.reject_recommendation import RejectRecommendation
from intelligence.gpt_wrapper import GPTWrapper
from state.constraints.constraints_extractor import ConstraintsExtractor
from state.common_state_manager import CommonStateManager
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from state.message import Message
import time
from domain_specific_config_loader import DomainSpecificConfigLoader
import os
from dotenv import load_dotenv
import yaml


class TestUserIntentsClassifier:
    """
    A test suite for the MultilabelUserIntentsClassifier, designed to verify its functionality using the pytest framework.

    This class reads pairs of input messages and expected intents from a CSV file and uses these pairs to test the MultilabelUserIntentsClassifier. 
    The test checks whether the length of the returned result matches the expected length, and whether the returned intents match the expected intents.
    """
    @pytest.mark.parametrize("input_message, expected_intent_1, expected_intent_2, leng",
                             [(row['Input'], row['Output1'], row['Output2'], row['leng']) for index, row in pd.read_csv('test/clothing_user_intent_test.csv', encoding='ISO-8859-1').iterrows()])
    def test_multilabel_user_intents_classifier(self, input_message, expected_intent_1, expected_intent_2, leng):
        gpt_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
        with open('system_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        ask_for_recommendation = AskForRecommendation(config)

        possible_goals = {ask_for_recommendation}
        domain_specific_config_loader = DomainSpecificConfigLoader()
        domain = domain_specific_config_loader.load_domain()

        inquire_classification_fewshots = domain_specific_config_loader.load_inquire_classification_fewshots()
        accept_classification_fewshots = domain_specific_config_loader.load_accept_classification_fewshots()
        reject_classification_fewshots = domain_specific_config_loader.load_reject_classification_fewshots()

        user_intents = [Inquire(None, inquire_classification_fewshots,domain,config),
                        AcceptRecommendation(
                            None,None, accept_classification_fewshots, domain,config),
                        RejectRecommendation(None,None, reject_classification_fewshots, domain,config)]
        classifier = MultilabelUserIntentsClassifier(user_intents, gpt_wrapper,config)
        message = Message("user", input_message)
        state = CommonStateManager(possible_goals)
        state.update_conv_history(message)
        result = classifier.classify(state)
        #time.sleep(2)
        assert len(result) == leng
        
        if leng == 1:
            assert result[0].__class__.__name__ == expected_intent_1
        elif leng == 2:
            assert set([result[0].__class__.__name__, result[1].__class__.__name__]) == set(
                [expected_intent_1, expected_intent_2])
        