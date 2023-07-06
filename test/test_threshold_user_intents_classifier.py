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


class TestThresholdUserIntentsClassifier:
    """
    A test suite for the MultilabelUserIntentsClassifier, designed to verify its functionality using the pytest framework.

    This class reads pairs of input messages and expected intents from a CSV file and uses these pairs to test the MultilabelUserIntentsClassifier. 
    The test checks whether the length of the returned result matches the expected length, and whether the returned intents match the expected intents.
    """
    @pytest.mark.parametrize("input_message, expected_intent_1, expected_intent_2, leng",
                             [(row['Input'], row['Output1'], row['Output2'], row['leng']) for index, row in pd.read_csv('test/Threshold_User_Intent_Test.csv', encoding='ISO-8859-1').iterrows()])
    def test_multilabel_user_intents_classifier(self, input_message, expected_intent_1, expected_intent_2, leng):
        gpt_wrapper = GPTWrapper()
        

        constraints_extractor = ConstraintsExtractor(gpt_wrapper)
        ask_for_recommendation = AskForRecommendation()

        curr_res_extractor = CurrentItemsExtractor(gpt_wrapper)

        possible_goals = {ask_for_recommendation}

        user_intents = [Inquire(curr_res_extractor),  AcceptRecommendation(None, curr_res_extractor), RejectRecommendation(None, curr_res_extractor)]

        classifier = MultilabelUserIntentsClassifier(user_intents, gpt_wrapper)
        message = Message("user", input_message)
        state = CommonStateManager(possible_goals)
        state.update_conv_history(message)
        result = classifier.classify(state)
        time.sleep(2)
        assert len(result) == leng
        
        if leng == 1:
            assert result[0].__class__.__name__ == expected_intent_1
        elif leng == 2:
            assert set([result[0].__class__.__name__, result[1].__class__.__name__]) == set(
                [expected_intent_1, expected_intent_2])
        