import logging.config
import warnings

import openai.error
import yaml

from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper

from intelligence.gpt_wrapper import GPTWrapper
from intelligence.gpt_wrapper_observer import GPTWrapperObserver
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
from information_retrievers.neural_information_retriever import NeuralInformationRetriever
from information_retrievers.filter.check_location import CheckLocation
from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.neural_ir.neural_search_engine import NeuralSearchEngine
from information_retrievers.neural_ir.statics import *
from information_retrievers.neural_ir.neural_embedder import BERT_model
from user_intent.reject_recommendation import RejectRecommendation
from information_retrievers.data_holder import DataHolder
from state.constraints.three_steps_constraints_updater import ThreeStepsConstraintsUpdater
from domain_specific_config_loader import DomainSpecificConfigLoader


class ConvRecSystem(GPTWrapperObserver):
    """
    Class responsible for setting up and running the conversational recommendation system.

    :param config: storing how conv rec system should be setup
    """

    is_gpt_retry_notified: bool
    user_interface: UserInterface
    dialogue_manager: DialogueManager

    def __init__(self, config: dict, user_defined_constraint_mergers: list, user_constraint_status_objects: list):
        domain_specific_config_loader = DomainSpecificConfigLoader()        
        domain = domain_specific_config_loader.load_domain()
        
        # TEMP
        constraints = config['ALL_CONSTRAINTS']
        self._constraints = constraints
        geocoder_wrapper = GoogleV3Wrapper()
        
        model = config["MODEL"]
        llm_wrapper = GPTWrapper(model_name=model, observers=[self])
        
        # Constraints
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()
        
        #TODO: generalize
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
        #TODO: generalize
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
        check_location = CheckLocation(
            config['DEFAULT_MAX_DISTANCE_IN_KM'], config['DISTANCE_TYPE'])
        check_cuisine_type = CheckCuisineDishType()
        check_already_recommended_restaurant = CheckAlreadyRecommendedRestaurant()
        data_holder = DataHolder(config["PATH_TO_RESTAURANT_METADATA"], config["PATH_TO_RESTAURANT_REVIEW_EMBEDDINGS"],
                                 config["PATH_TO_RESTAURANT_REVIEW_EMBEDDING_MATRIX"],
                                 config["PATH_TO_NUM_OF_REVIEWS_PER_RESTAURANT"])
        filter_restaurant = FilterRestaurants(geocoder_wrapper, check_location, check_cuisine_type, check_already_recommended_restaurant,
                                              data_holder, config["FILTER_CONSTRAINTS"])

        #Initialize IR
        BERT_name = config["IR_BERT_MODEL_NAME"]
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        embedder = BERT_model(BERT_model_name, tokenizer_name, False)
        engine = NeuralSearchEngine(embedder)
        information_retriever = NeuralInformationRetriever(engine, data_holder)
        
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
                user_intents, llm_wrapper, True)
        else:
            user_intents_classifier = PromptBasedUserIntentsClassifier(
                user_intents, llm_wrapper)
        
        # Initialize State
        state = CommonStateManager(
            {AskForRecommendation(config), user_intents[0], user_intents[2], user_intents[3]}, AskForRecommendation(config))
        state.update("unsatisfied_goals", [
            {"user_intent": AskForRecommendation(config), "utterance_index": 0}])
        
        # Initialize Rec Action
        rec_actions = [Answer(config, llm_wrapper, filter_restaurant, information_retriever, domain),
                       ExplainPreference(),
                       Recommend(llm_wrapper, filter_restaurant, information_retriever, domain, user_constraint_status_objects,
                                 config, constraints_categories),
                       RequestInformation(user_constraint_status_objects, constraints_categories), PostRejectionAction(),
                       PostAcceptanceAction()]
        

        rec_action_classifier = CommonRecActionsClassifier(rec_actions)
        
        # Initialize system
        self.user_interface = Terminal()
        self.dialogue_manager = DialogueManager(
            state, user_intents_classifier, rec_action_classifier, llm_wrapper)
        self.is_gpt_retry_notified = False
        
        

    def run(self) -> None:
        """
        Run the conv rec system.
        """
        self.user_interface.display_to_user()
        while True:
            user_input = self.user_interface.get_user_input("User: ")
            if user_input == 'quit' or user_input == 'q':
                break
            response = self.dialogue_manager.get_response(user_input)
            self.user_interface.display_to_user(f'Recommender: {response}')
            self.is_gpt_retry_notified = False

    def notify_gpt_retry(self, retry_info: dict) -> None:
        """
        Notify this object that gpt re-requested due to Rate Limit Error.

        :param retry_info: dictionary that contains information about retry
        """
        if not self.is_gpt_retry_notified:
            if isinstance(retry_info.get('output'), openai.error.ServiceUnavailableError) or \
                    isinstance(retry_info.get('output'), openai.error.APIConnectionError):
                self.user_interface.display_to_user(
                    "There were some issues with the OpenAI server. It might take longer than usual.")
            else:
                self.user_interface.display_to_user(
                    "OpenAI API are currently busy. It might take longer than usual.")

        self.is_gpt_retry_notified = True