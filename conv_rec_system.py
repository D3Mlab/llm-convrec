import openai.error

from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from information_retrievers.item.item_loader import ItemLoader

from intelligence.gpt_wrapper import GPTWrapper
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper
from warning_observer import WarningObserver
from rec_action.answer import Answer
from rec_action.explain_preference import ExplainPreference
from rec_action.recommend import Recommend
from rec_action.request_information import RequestInformation
from rec_action.post_acceptance_action import PostAcceptanceAction
from rec_action.post_rejection_action import PostRejectionAction
from dialogue_manager import DialogueManager
from state.common_state_manager import CommonStateManager
from state.constraints.one_step_constraints_updater import OneStepConstraintsUpdater
from state.constraints.safe_constraints_remover import SafeConstraintsRemover
from user.terminal import Terminal
from user.gradio import GradioInterface
from user.user_interface import UserInterface
from user_intent.accept_recommendation import AcceptRecommendation
from user_intent.ask_for_recommendation import AskForRecommendation
from state.constraints.constraints_classifier import ConstraintsClassifier
from state.constraints.constraints_remover import ConstraintsRemover
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from state.constraints.key_value_pair_constraints_extractor import KeyValuePairConstraintsExtractor
from user_intent.extractors.rejected_items_extractor import RejectedItemsExtractor
from user_intent.inquire import Inquire
from user_intent.provide_preference import ProvidePreference
from user_intent.classifiers.prompt_based_user_intents_classifier import PromptBasedUserIntentsClassifier
from user_intent.classifiers.multilabel_user_intents_classifier import MultilabelUserIntentsClassifier
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from rec_action.common_rec_actions_classifier import CommonRecActionsClassifier
from information_retrievers.embedder.statics import *
from information_retrievers.embedder.bert_embedder import BERT_model
from user_intent.reject_recommendation import RejectRecommendation
from state.constraints.three_steps_constraints_updater import ThreeStepsConstraintsUpdater
from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retrievers.search_engine.pd_search_engine import PDSearchEngine
from information_retrievers.search_engine.vector_database_search_engine import VectorDatabaseSearchEngine
from information_retrievers.metadata_wrapper import MetadataWrapper
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.filter.filter import Filter
from information_retrievers.information_retrieval import InformationRetrieval
from rec_action.response_type.recommend_hard_coded_based_resp import RecommendHardCodedBasedResponse
from rec_action.response_type.recommend_prompt_based_resp import RecommendPromptBasedResponse
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from rec_action.response_type.request_information_hard_coded_resp import RequestInformationHardCodedBasedResponse
from rec_action.response_type.accept_hard_code_resp import AcceptHardCodedBasedResponse
from rec_action.response_type.reject_hard_code_resp import RejectHardCodedBasedResponse

class ConvRecSystem(WarningObserver):
    """
    Class responsible for setting up and running the conversational recommendation system.

    :param config: storing how conv rec system should be setup
    """

    is_gpt_retry_notified: bool
    user_interface: UserInterface
    dialogue_manager: DialogueManager

    def __init__(self, config: dict, openai_api_key_or_gradio_url: str,
                 user_defined_constraint_mergers: list = None,
                 user_constraint_status_objects: list = None,
                 user_defined_filter: list[Filter] = None,
                 user_interface_str: str = None):
        
        domain_specific_config_loader = DomainSpecificConfigLoader()
        domain = domain_specific_config_loader.load_domain()

        model = config["MODEL"]

        if not isinstance(openai_api_key_or_gradio_url, str):
            raise TypeError("The variable type of OPENAI_API_KEY or GRADIO_URL is wrong.")

        if config['LLM'] == "Alpaca Lora":
            llm_wrapper = AlpacaLoraWrapper(openai_api_key_or_gradio_url)
        else:
            llm_wrapper = GPTWrapper(openai_api_key_or_gradio_url, model_name=model, observers=[self])

        hard_coded_responses = domain_specific_config_loader.load_hard_coded_responses()

        # Constraints
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()

        #TODO: generalize 3 step constraints updater
        if config['CONSTRAINTS_UPDATER'] == "three_steps_constraints_updater":
            constraints_extractor = KeyValuePairConstraintsExtractor(
                llm_wrapper, constraints_categories, config)
            constraints_classifier = ConstraintsClassifier(
                llm_wrapper, constraints_categories, config)
            if config['ENABLE_CONSTRAINTS_REMOVAL']:
                constraints_remover = ConstraintsRemover(
                    llm_wrapper, constraints_categories, config)
            else:
                constraints_remover = None
            
            constraints_updater = ThreeStepsConstraintsUpdater(
                constraints_extractor, constraints_classifier, constraints_categories,
                constraints_remover=constraints_remover)
        
        elif config['CONSTRAINTS_UPDATER'] == "safe_three_steps_constraints_updater":
            constraints_extractor = KeyValuePairConstraintsExtractor(
                llm_wrapper, constraints_categories, config)
            constraints_classifier = ConstraintsClassifier(
                llm_wrapper, constraints_categories, config)
            if config['ENABLE_CONSTRAINTS_REMOVAL']:
                constraints_remover = SafeConstraintsRemover(
                    llm_wrapper, constraints_categories, config)
            else:
                constraints_remover = None
            
            constraints_updater = ThreeStepsConstraintsUpdater(
                constraints_extractor, constraints_classifier, constraints_categories,
                constraints_remover=constraints_remover)
       
        else:
            if config['LLM'] == "alpaca lora":
                temperature_zero_llm_wrapper = AlpacaLoraWrapper(openai_api_key_or_gradio_url, temperature=0)
            else:
                temperature_zero_llm_wrapper = GPTWrapper(
                    openai_api_key_or_gradio_url, model_name=model, temperature=0, observers=[self])

            constraints_updater = OneStepConstraintsUpdater(temperature_zero_llm_wrapper,
                                                            constraints_categories,
                                                            constraints_fewshots, domain,
                                                            user_defined_constraint_mergers, config)
        # Initialize Extractors
        accepted_items_fewshots = domain_specific_config_loader.load_rejected_items_fewshots()
        rejected_items_fewshots = domain_specific_config_loader.load_accepted_items_fewshots()
        curr_items_fewshots = domain_specific_config_loader.load_current_items_fewshots()
        
        accepted_items_extractor = AcceptedItemsExtractor(
            llm_wrapper, domain, accepted_items_fewshots, config)
        rejected_items_extractor = RejectedItemsExtractor(
            llm_wrapper, domain, rejected_items_fewshots, config)
        curr_items_extractor = CurrentItemsExtractor(llm_wrapper, domain, curr_items_fewshots, config)

        # Initialize Filters
        metadata_wrapper = MetadataWrapper()
        filter_item = FilterApplier(metadata_wrapper)
        if user_defined_filter:
            filter_item.filters.extend(user_defined_filter)

        BERT_name = config["IR_BERT_MODEL_NAME"]
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        embedder = BERT_model(BERT_model_name, tokenizer_name, False)
        if config['SEARCH_ENGINE'] == "pandas":
            search_engine = PDSearchEngine(embedder)
        else:
            search_engine = VectorDatabaseSearchEngine(embedder)
        information_retrieval = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())
        
        # Initialize User Intent
        inquire_classification_fewshots = domain_specific_config_loader.load_inquire_classification_fewshots()
        accept_classification_fewshots = domain_specific_config_loader.load_accept_classification_fewshots()
        reject_classification_fewshots = domain_specific_config_loader.load_reject_classification_fewshots()

        user_intents = [Inquire(curr_items_extractor, inquire_classification_fewshots,domain, config),
                        ProvidePreference(constraints_updater, curr_items_extractor, user_constraint_status_objects, config),
                        AcceptRecommendation(
                            accepted_items_extractor, curr_items_extractor, accept_classification_fewshots, domain, config),
                        RejectRecommendation(rejected_items_extractor, curr_items_extractor, reject_classification_fewshots, domain, config)]

        if config["USER_INTENTS_CLASSIFIER"] == "MultilabelUserIntentsClassifier":
            user_intents_classifier = MultilabelUserIntentsClassifier(
                user_intents, llm_wrapper, config, True)
        else:
            user_intents_classifier = PromptBasedUserIntentsClassifier(
                user_intents, llm_wrapper)
        
        # Initialize State
        state = CommonStateManager(
            {AskForRecommendation(config), user_intents[0], user_intents[2], user_intents[3]}, AskForRecommendation(config))
        state.update("unsatisfied_goals", [
            {"user_intent": AskForRecommendation(config), "utterance_index": 0}])
        
        # Initialize Rec Action
        recc_resp = RecommendPromptBasedResponse(llm_wrapper, filter_item, information_retrieval, domain, hard_coded_responses,  config, observers=[self])
 
        if config['RECOMMEND_RESPONSE_TYPE'] == 'hard coded':
            recc_resp = RecommendHardCodedBasedResponse(llm_wrapper, filter_item, information_retrieval, domain, config, hard_coded_responses)
        
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, filter_item, information_retrieval, domain, hard_coded_responses,observers=[self])
        requ_info_resp = RequestInformationHardCodedBasedResponse(hard_coded_responses)
        accept_resp = AcceptHardCodedBasedResponse(hard_coded_responses)
        reject_resp = RejectHardCodedBasedResponse(hard_coded_responses)


        rec_actions = [Answer(answer_resp),
                       ExplainPreference(),
                       Recommend(user_constraint_status_objects, hard_coded_responses, recc_resp, config),
                       RequestInformation(user_constraint_status_objects, hard_coded_responses, requ_info_resp), 
                       PostRejectionAction(reject_resp),
                       PostAcceptanceAction(accept_resp)]

        rec_action_classifier = CommonRecActionsClassifier(rec_actions)

        # Initialize system
        if user_interface_str == "demo":
            self.user_interface = GradioInterface()
        else:
            self.user_interface = Terminal()

        self.dialogue_manager = DialogueManager(
            state, user_intents_classifier, rec_action_classifier, llm_wrapper, hard_coded_responses)
        self.is_gpt_retry_notified = False
        self.is_warning_notified = False
        self.init_msg = f'Recommender: Hello there! I am a {domain} recommender. Please provide me with some preferences for what you are looking for. For example, {constraints_categories[0]["key"]}, {constraints_categories[1]["key"]}, or {constraints_categories[2]["key"]}. Thanks!'


    def run(self) -> None:
        """
        Run the conv rec system.
        """
        self.user_interface.display_to_user(self.init_msg)
        while True:
            user_input = self.user_interface.get_user_input("User: ")
            if user_input == 'quit' or user_input == 'q':
                break
            response = self.get_response(user_input)
            self.user_interface.display_to_user(f'Recommender: {response}')

    def notify_gpt_retry(self, retry_info: dict) -> None:
        """
        Notify this object that gpt re-requested due to Rate Limit Error.

        :param retry_info: dictionary that contains information about retry
        """
        if not self.is_gpt_retry_notified:
            if isinstance(retry_info.get('output'), openai.error.ServiceUnavailableError) or \
                    isinstance(retry_info.get('output'), openai.error.APIConnectionError):
                self.user_interface.display_warning(
                    "There were some issues with the OpenAI server. It might take longer than usual.")
            else:
                self.user_interface.display_warning(
                    "OpenAI API are currently busy. It might take longer than usual.")

        self.is_gpt_retry_notified = True
        
    def get_response(self, user_input: str) -> str:
        """
        Respond to the user input
        """
        self.is_gpt_retry_notified = False
        self.is_warning_notified = False
        return self.dialogue_manager.get_response(user_input)

    def notify_warning(self):
        """
        Notify this object about warnings.
        """
        if not self.is_warning_notified:
            self.user_interface.display_warning(
                "Sorry.. running into some difficulties, this is going to take longer than usual.")
        self.is_warning_notified = True
