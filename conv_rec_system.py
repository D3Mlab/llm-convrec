import openai.error

from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from information_retrievers.item_loader import ItemLoader

from intelligence.gpt_wrapper import GPTWrapper
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
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from information_retrievers.filter import Filter
from information_retrievers.information_retrieval import InformationRetrieval


class ConvRecSystem(WarningObserver):
    """
    Class responsible for setting up and running the conversational recommendation system.

    :param config: storing how conv rec system should be setup
    """

    is_gpt_retry_notified: bool
    user_interface: UserInterface
    dialogue_manager: DialogueManager

    def __init__(self, config: dict, user_defined_constraint_mergers: list,
                 user_interface_str: str=None):
        constraints = config['ALL_CONSTRAINTS']
        
        domain_specific_config_loader = DomainSpecificConfigLoader()
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()
        domain = domain_specific_config_loader.load_domain()
        
        model = config["MODEL"]


        self._constraints = constraints
        specific_location_required = config["SPECIFIC_LOCATION_REQUIRED"]
        # TEMP
        geocoder_wrapper = GoogleV3Wrapper()
         
        llm_wrapper = GPTWrapper(model_name=model, observers=[self])
        curr_restaurant_extractor = CurrentItemsExtractor(llm_wrapper, domain)
        if config['CONSTRAINTS_UPDATER'] == "three_steps_constraints_updater":
            constraints_extractor = KeyValuePairConstraintsExtractor(
                llm_wrapper, constraints)
            constraints_classifier = ConstraintsClassifier(
                llm_wrapper, constraints)
            if config['ENABLE_CONSTRAINTS_REMOVAL']:
                constraints_remover = ConstraintsRemover(
                    llm_wrapper, constraints)
            else:
                constraints_remover = None
            constraints_updater = ThreeStepsConstraintsUpdater(
                constraints_extractor, constraints_classifier, geocoder_wrapper,
                constraints_remover=constraints_remover,
                cumulative_constraints=set(config['CUMULATIVE_CONSTRAINTS']),
                enable_location_merge=config['ENABLE_LOCATION_MERGE'])
        elif config['CONSTRAINTS_UPDATER'] == "safe_three_steps_constraints_updater":
            constraints_extractor = KeyValuePairConstraintsExtractor(
                llm_wrapper, constraints)
            constraints_classifier = ConstraintsClassifier(
                llm_wrapper, constraints)
            if config['ENABLE_CONSTRAINTS_REMOVAL']:
                constraints_remover = SafeConstraintsRemover(
                    llm_wrapper, default_keys=constraints)
            else:
                constraints_remover = None
            constraints_updater = ThreeStepsConstraintsUpdater(
                constraints_extractor, constraints_classifier, geocoder_wrapper,
                constraints_remover=constraints_remover,
                cumulative_constraints=set(config['CUMULATIVE_CONSTRAINTS']),
                enable_location_merge=config['ENABLE_LOCATION_MERGE'])
        else:
            temperature_zero_llm_wrapper = GPTWrapper(temperature=0)
            constraints_updater = OneStepConstraintsUpdater(temperature_zero_llm_wrapper,
                                                            constraints_categories,
                                                            constraints_fewshots, domain,
                                                            user_defined_constraint_mergers)
        accepted_restaurants_extractor = AcceptedItemsExtractor(
            llm_wrapper, domain)
        rejected_restaurants_extractor = RejectedItemsExtractor(
            llm_wrapper, domain)

        checkers = domain_specific_config_loader.load_checkers()
        metadata_wrapper = MetadataWrapper()
        filter_item = Filter(metadata_wrapper, checkers)
        BERT_name = config["BERT_MODEL_NAME"]
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        embedder = BERT_model(BERT_model_name, tokenizer_name, False)
        if config['SEARCH_ENGINE'] == "pandas":
            search_engine = PDSearchEngine(embedder)
        else:
            search_engine = VectorDatabaseSearchEngine(embedder)
        information_retrieval = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())

        default_location = config.get('DEFAULT_LOCATION') if config.get(
            'DEFAULT_LOCATION') != 'None' else None
        

        inquire_classification_fewshots = domain_specific_config_loader.load_inquire_classification_fewshots()
        accept_classification_fewshots = domain_specific_config_loader.load_accept_classification_fewshots()
        reject_classification_fewshots = domain_specific_config_loader.load_reject_classification_fewshots()

        user_intents = [Inquire(curr_restaurant_extractor, inquire_classification_fewshots,domain),
                        ProvidePreference(constraints_updater, curr_restaurant_extractor, geocoder_wrapper,
                                          default_location=default_location),
                        AcceptRecommendation(
                            accepted_restaurants_extractor, curr_restaurant_extractor, accept_classification_fewshots, domain),
                        RejectRecommendation(rejected_restaurants_extractor, curr_restaurant_extractor, reject_classification_fewshots, domain)]

        if user_interface_str == "demo":
            self.user_interface = GradioInterface()
        else:
            self.user_interface = Terminal()

        rec_actions = [Answer(config, llm_wrapper, filter_item, information_retrieval, domain, observers=[self]),
                       ExplainPreference(),
                       Recommend(llm_wrapper, filter_item, information_retrieval, domain, observers=[self],
                                 mandatory_constraints=config['MANDATORY_CONSTRAINTS'],
                                 specific_location_required=specific_location_required),
                       RequestInformation(mandatory_constraints=config['MANDATORY_CONSTRAINTS'],
                                          specific_location_required=specific_location_required), PostRejectionAction(),
                       PostAcceptanceAction()]
        if config["USER_INTENTS_CLASSIFIER"] == "MultilabelUserIntentsClassifier":
            user_intents_classifier = MultilabelUserIntentsClassifier(
                user_intents, llm_wrapper, True)
        else:
            user_intents_classifier = PromptBasedUserIntentsClassifier(
                user_intents, llm_wrapper)

        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        state = CommonStateManager(
            {AskForRecommendation(), user_intents[0], user_intents[2], user_intents[3]}, AskForRecommendation())
        state.update("unsatisfied_goals", [
            {"user_intent": AskForRecommendation(), "utterance_index": 0}])
        self.dialogue_manager = DialogueManager(
            state, user_intents_classifier, rec_action_classifier, llm_wrapper)
        self.is_gpt_retry_notified = False
        self.is_warning_notified = False

    def run(self) -> None:
        """
        Run the conv rec system.
        """
        init_msg = f'Recommender: Hello there! I am a restaurant recommender. Please provide me with some preferences for what you are looking for. For example, {self._constraints[1]}, {self._constraints[2]}, or {self._constraints[3]}. Thanks!'
        self.user_interface.display_to_user(init_msg)
        while True:
            user_input = self.user_interface.get_user_input("User: ")
            if user_input == 'quit' or user_input == 'q':
                break
            response = self.dialogue_manager.get_response(user_input)
            self.user_interface.display_to_user(f'Recommender: {response}')
            self.is_gpt_retry_notified = False
            self.is_warning_notified = False

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

    def notify_warning(self):
        """
        Notify this object about warnings.
        """
        if not self.is_warning_notified:
            self.user_interface.display_warning(
                "Sorry.. running into some difficulties, this is going to take longer than usual.")
        self.is_warning_notified = True
