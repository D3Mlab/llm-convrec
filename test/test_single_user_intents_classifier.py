import pytest
import pandas as pd
from user_intent.classifiers.multilabel_user_intents_classifier import MultilabelUserIntentsClassifier
from user_intent.ask_for_recommendation import AskForRecommendation
from user_intent.inquire import Inquire
from user_intent.provide_preference import ProvidePreference
from user_intent.accept_recommendation import AcceptRecommendation
from user_intent.reject_recommendation import RejectRecommendation
from intelligence.gpt_wrapper import GPTWrapper
from state.constraints.constraints_extractor import ConstraintsExtractor
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from user_intent.extractors.rejected_items_extractor import RejectedItemsExtractor
from state.common_state_manager import CommonStateManager
from state.message import Message
import time


class TestSingleUserIntentsClassifier:
    """
    A test suite for the MultilabelUserIntentsClassifier. This class uses the pytest framework to perform unit testing on the MultilabelUserIntentsClassifier.

    The test suite reads from a CSV file to provide pairs of input messages and expected intents. It classifies the message using the classifier and checks if the first returned result matches the expected intent.
    """

    @pytest.mark.parametrize("input_message, expected_intent",
                             [(row['Input'], row['Output1']) for index, row in pd.read_csv('test/Singleuserintenttest.csv', encoding='ISO-8859-1').iterrows()])
    def test_multilabel_user_intents_classifier(self, input_message, expected_intent):
        gpt_wrapper = GPTWrapper()

        constraints_extractor = ConstraintsExtractor()
        curr_res_extractor = CurrentItemsExtractor(gpt_wrapper)
        ask_for_recommendation = AskForRecommendation()

        possible_goals = {ask_for_recommendation}
        user_intents = [AskForRecommendation(), Inquire(curr_res_extractor
                                                        ), ProvidePreference(constraints_extractor, None, curr_res_extractor), AcceptRecommendation(AcceptedItemsExtractor, curr_res_extractor), RejectRecommendation(RejectedItemsExtractor, curr_res_extractor)]
        classifier = MultilabelUserIntentsClassifier(user_intents, gpt_wrapper)
        message = Message("user", input_message)
        state = CommonStateManager(possible_goals)
        state.update_conv_history(message)
        result = classifier.classify(state)
        time.sleep(8)
        assert result[0].__class__.__name__ == expected_intent
