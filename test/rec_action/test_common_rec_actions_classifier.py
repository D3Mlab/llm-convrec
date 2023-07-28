from rec_action.answer import Answer
from rec_action.common_rec_actions_classifier import CommonRecActionsClassifier
from rec_action.explain_preference import ExplainPreference
from rec_action.recommend import Recommend
from rec_action.request_information import RequestInformation
from state.common_state_manager import CommonStateManager
from state.message import Message
from user_intent.ask_for_recommendation import AskForRecommendation
from user_intent.inquire import Inquire
from rec_action.post_acceptance_action import PostAcceptanceAction
from rec_action.post_rejection_action import PostRejectionAction
from rec_action.response_type.recommend_prompt_based_resp import RecommendPromptBasedResponse
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from rec_action.response_type.request_information_hard_coded_resp import RequestInformationHardCodedBasedResponse
from rec_action.response_type.accept_hard_code_resp import AcceptHardCodedBasedResponse
from rec_action.response_type.reject_hard_code_resp import RejectHardCodedBasedResponse
from intelligence.gpt_wrapper import GPTWrapper
from domain_specific_config_loader import DomainSpecificConfigLoader
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

with open("system_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

llm_wrapper = GPTWrapper(os.environ['OPENAI_API_KEY'])
domain_specific_config_loader = DomainSpecificConfigLoader(config)
domain = "restaurants"


class TestCommonRecActionsClassifier:
    """
    Test class for RecActionsClassifier
    """

    def test_classify_top_one_request_information(self) -> None:
        """
        Test whether CommonRecActionsClassifier classifies recommender action to RequestInformation
        when user asks for recommendation, but it doesn't have the mandatory constraints and k = 1.
        """
        recc_resp = RecommendPromptBasedResponse(None, None, None, domain,
                                                 [], config, [])

        answer_resp = AnswerPromptBasedResponse(config, None, None, None, domain,
                                                [],  [], [], [], [])
        requ_info_resp = RequestInformationHardCodedBasedResponse([], [])
        accept_resp = AcceptHardCodedBasedResponse([])
        reject_resp = RejectHardCodedBasedResponse([])

        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend([], [], recc_resp),
                       RequestInformation([], [], requ_info_resp),
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]
        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(
            Message("user", "I want recommendation in Toronto"))
        state_manager.update("hard_constraints", {"location": "Toronto"})
        state_manager.update("unsatisfied_goals", [
                             {"user_intent": AskForRecommendation(config), "utterance_index": 0}])

        result = rec_action_classifier.classify(state_manager, k=1)

        assert len(result) == 1 and isinstance(result[0], RequestInformation)

    def test_classify_top_one_recommend(self) -> None:
        """
        Test whether CommonRecActionsClassifier classifies recommender action to Recommend
        when user asks for recommendation with mandatory constraints filled and k = 1.
        """
        recc_resp = RecommendPromptBasedResponse(None, None, None, domain,
                                                 [], config, [])

        answer_resp = AnswerPromptBasedResponse(config, None, None, None, domain,
                                                [], [], [], [], [])
        requ_info_resp = RequestInformationHardCodedBasedResponse([], [])
        accept_resp = AcceptHardCodedBasedResponse([])
        reject_resp = RejectHardCodedBasedResponse([])

        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend([], [], recc_resp),
                       RequestInformation([], [], requ_info_resp),
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]
        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(
            Message("user", "I want italian restaurants in Toronto"))
        state_manager.update("hard_constraints", {
                             "location": "Toronto", "cuisine type": "italian"})
        state_manager.update("unsatisfied_goals", [
                             {"user_intent": AskForRecommendation(config), "utterance_index": 0}])

        result = rec_action_classifier.classify(state_manager, k=1)

        assert len(result) == 1 and isinstance(result[0], Recommend)

    def test_classify_top_one_answer(self) -> None:
        """
        Test whether RecActionsClassifiers classifies recommender action to Answer
        when unsatisfied goal has user intent, Inquire, and k = 1.
        """
        recc_resp = RecommendPromptBasedResponse(None, None, None, domain,
                                                 [], config, [])

        answer_resp = AnswerPromptBasedResponse(config, None, None, None, domain,
                                                [], [], [], [], [])
        requ_info_resp = RequestInformationHardCodedBasedResponse([], [])
        accept_resp = AcceptHardCodedBasedResponse([])
        reject_resp = RejectHardCodedBasedResponse([])

        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend([], [], recc_resp),
                       RequestInformation([], [], requ_info_resp),
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]
        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(
            Message("user", "Do they have patio?"))
        state_manager.update("unsatisfied_goals", [
                             {"user_intent": Inquire(None, None, domain, config), "utterance_index": 0}])

        result = rec_action_classifier.classify(state_manager, k=1)

        assert len(result) == 1 and isinstance(result[0], Answer)

    def test_classify_top_no_rec_action(self) -> None:
        """
        Test whether RecActionsClassifiers doesn't classify to any recommender action when unsatisfied goals
        doesn't exist
        """
        recc_resp = RecommendPromptBasedResponse(None, None, None, domain,
                                                 [], config, [])
        answer_resp = AnswerPromptBasedResponse(config, None, None, None, domain,
                                                [], [], [], [], [])

        requ_info_resp = RequestInformationHardCodedBasedResponse([], [])
        accept_resp = AcceptHardCodedBasedResponse([])
        reject_resp = RejectHardCodedBasedResponse([])

        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend([], [], recc_resp),
                       RequestInformation([], [], requ_info_resp),
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]
        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message("user", "Hello"))

        result = rec_action_classifier.classify(state_manager, k=1)

        assert len(result) == 0

    def test_classify_two_actions(self) -> None:
        """
        Test whether CommonRecActionsClassifier two recommender actions correctly by prioritizing more recent
        unsatisfied goals.
        """
        recc_resp = RecommendPromptBasedResponse(None, None, None, domain,
                                                 [], config, [])

        answer_resp = AnswerPromptBasedResponse(config, None, None, None, domain,
                                                [], [], [], [], [])

        requ_info_resp = RequestInformationHardCodedBasedResponse([], [])
        accept_resp = AcceptHardCodedBasedResponse([])
        reject_resp = RejectHardCodedBasedResponse([])

        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend([], [], recc_resp),
                       RequestInformation([], [], requ_info_resp),
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]
        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state_manager = CommonStateManager(set())
        state_manager.update("conv_history", [Message("DUMMY", "DUMMY")] * 4)
        state_manager.update("unsatisfied_goals", [{"user_intent": AskForRecommendation(config), "utterance_index": 0},
                                                   {"user_intent": Inquire(None, None, domain, config), "utterance_index": 4}])
        result = rec_action_classifier.classify(state_manager, k=5)
        assert len(result) == 2 and result == [rec_actions[3], rec_actions[0]]
