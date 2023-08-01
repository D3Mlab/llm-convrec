import openai.error

from information_retriever.item.item_loader import ItemLoader

from intelligence.gpt_wrapper import GPTWrapper
from warning_observer import WarningObserver
from rec_action.answer import Answer
from rec_action.recommend import Recommend
from rec_action.request_information import RequestInformation
from rec_action.post_acceptance_action import PostAcceptanceAction
from rec_action.post_rejection_action import PostRejectionAction
from dialogue_manager import DialogueManager
from state.common_state_manager import CommonStateManager
from state.constraints.one_step_constraints_updater import OneStepConstraintsUpdater
from user.terminal import Terminal
from user.gradio import GradioInterface
from user.user_interface import UserInterface
from user_intent.accept_recommendation import AcceptRecommendation
from user_intent.ask_for_recommendation import AskForRecommendation
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from user_intent.extractors.rejected_items_extractor import RejectedItemsExtractor
from user_intent.inquire import Inquire
from user_intent.provide_preference import ProvidePreference
from user_intent.classifiers.multilabel_user_intents_classifier import MultilabelUserIntentsClassifier
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from rec_action.common_rec_actions_classifier import CommonRecActionsClassifier
from information_retriever.embedder.statics import *
from information_retriever.embedder.bert_embedder import BERT_model
from user_intent.reject_recommendation import RejectRecommendation
from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retriever.search_engine.matmul_search_engine import MatMulSearchEngine
from information_retriever.search_engine.vector_database_search_engine import VectorDatabaseSearchEngine
from information_retriever.metadata_wrapper import MetadataWrapper
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.filter.filter import Filter
from information_retriever.information_retrieval import InformationRetrieval
from rec_action.response_type.recommend_prompt_based_resp import RecommendPromptBasedResponse
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from rec_action.response_type.request_information_hard_coded_resp import RequestInformationHardCodedBasedResponse
from rec_action.response_type.accept_hard_code_resp import AcceptHardCodedBasedResponse
from rec_action.response_type.reject_hard_code_resp import RejectHardCodedBasedResponse


class ConvRecSystem(WarningObserver):
    """
    Class responsible for setting up and running the conversational recommendation system.

    :param config: storing how conv rec system should be setup
    :param openai_api_key_or_gradio_url: api key for Open AI used to run ChatGPT or gradio URL used to run Alpaca Lora
    :param user_defined_constraint_mergers: constraint merger created by the user
    :param user_constraint_status_objects: objects that keep tracks the status of the constraints
    :param user_defined_filter: filters defined by the user
    :param user_interface_str: string that determines which user interface to use
    """

    is_gpt_retry_notified: bool
    is_warning_notified: bool
    user_interface: UserInterface
    dialogue_manager: DialogueManager
    init_msg: str

    def __init__(self, config: dict, openai_api_key_or_gradio_url: str,
                 user_defined_constraint_mergers: list = None,
                 user_constraint_status_objects: list = None,
                 user_defined_filter: list[Filter] = None,
                 user_interface_str: str = None):
        if user_constraint_status_objects is None:
            user_constraint_status_objects = []
        if user_defined_constraint_mergers is None:
            user_defined_constraint_mergers = []
        domain_specific_config_loader = DomainSpecificConfigLoader(config)
        domain = domain_specific_config_loader.load_domain()

        model = config["MODEL"]

        if not isinstance(openai_api_key_or_gradio_url, str):
            raise TypeError("The variable type of OPENAI_API_KEY or GRADIO_URL is wrong.")

        llm_wrapper = GPTWrapper(openai_api_key_or_gradio_url, model_name=model, observers=[self])

        hard_coded_responses = domain_specific_config_loader.load_hard_coded_responses()

        # Initialize Constraints related objects
        constraints_categories = domain_specific_config_loader.load_constraints_categories()
        constraints_fewshots = domain_specific_config_loader.load_constraints_updater_fewshots()

        temperature_zero_llm_wrapper = GPTWrapper(
            openai_api_key_or_gradio_url, model_name=model, temperature=0, observers=[self])
        constraints_updater = OneStepConstraintsUpdater(temperature_zero_llm_wrapper,
                                                        constraints_categories,
                                                        constraints_fewshots, domain,
                                                        user_defined_constraint_mergers, config)

        # Initialize Extractors
        accepted_items_fewshots = domain_specific_config_loader.load_accepted_items_fewshots()
        rejected_items_fewshots = domain_specific_config_loader.load_rejected_items_fewshots()
        curr_items_fewshots = domain_specific_config_loader.load_current_items_fewshots()
        
        accepted_items_extractor = AcceptedItemsExtractor(
            llm_wrapper, domain, accepted_items_fewshots, config)
        rejected_items_extractor = RejectedItemsExtractor(
            llm_wrapper, domain, rejected_items_fewshots, config)
        curr_items_extractor = CurrentItemsExtractor(llm_wrapper, domain, curr_items_fewshots, config)

        # Initialize Filters
        metadata_wrapper = MetadataWrapper(domain_specific_config_loader.load_item_metadata())
        filter_item = FilterApplier(metadata_wrapper, domain_specific_config_loader.load_filters())
        if user_defined_filter:
            filter_item.filters.extend(user_defined_filter)

        # Information Retrieval
        BERT_name = config["IR_BERT_MODEL_NAME"]
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        embedder = BERT_model(BERT_model_name, tokenizer_name)
        if config['SEARCH_ENGINE'] == "matmul":
            reviews_item_ids, reviews, reviews_embedding_matrix = \
                domain_specific_config_loader.load_data_for_pd_search_engine()
            search_engine = MatMulSearchEngine(embedder, reviews_item_ids, reviews, reviews_embedding_matrix, metadata_wrapper)
        else:
            reviews_item_ids, reviews, database = \
                domain_specific_config_loader.load_data_for_vector_database_search_engine()
            search_engine = VectorDatabaseSearchEngine(embedder, reviews_item_ids, reviews, database, metadata_wrapper)
        information_retrieval = InformationRetrieval(search_engine, metadata_wrapper, ItemLoader())
        
        # Initialize User Intent
        inquire_classification_fewshots = domain_specific_config_loader.load_inquire_classification_fewshots()
        accept_classification_fewshots = domain_specific_config_loader.load_accept_classification_fewshots()
        reject_classification_fewshots = domain_specific_config_loader.load_reject_classification_fewshots()

        user_intents = [Inquire(inquire_classification_fewshots,domain, config),
                        ProvidePreference(constraints_updater, user_constraint_status_objects, config),
                        AcceptRecommendation(
                            accepted_items_extractor, accept_classification_fewshots, domain, config),
                        RejectRecommendation(rejected_items_extractor, reject_classification_fewshots, domain, config)]

        user_intents_classifier = MultilabelUserIntentsClassifier(
            user_intents, llm_wrapper, config, True)

        
        # Initialize State
        state = CommonStateManager(
            {AskForRecommendation(config), user_intents[0], user_intents[2], user_intents[3]}, AskForRecommendation(config), current_items_extractor = curr_items_extractor)
        state.update("unsatisfied_goals", [
            {"user_intent": AskForRecommendation(config), "utterance_index": 0}])
        
        # Initialize Rec Action
        recc_resp = RecommendPromptBasedResponse(llm_wrapper, filter_item, information_retrieval, domain,
                                                 hard_coded_responses, config,
                                                 domain_specific_config_loader.load_constraints_categories(),
                                                 domain_specific_config_loader.load_explanation_metadata_blacklist(),
                                                 observers=[self])

        answer_resp = AnswerPromptBasedResponse(
            config, llm_wrapper, filter_item, information_retrieval, domain,
            hard_coded_responses,
            domain_specific_config_loader.load_answer_extract_category_fewshots(),
            domain_specific_config_loader.load_answer_ir_fewshots(),
            domain_specific_config_loader.load_answer_separate_questions_fewshots(),
            observers=[self]
        )
        requ_info_resp = RequestInformationHardCodedBasedResponse(hard_coded_responses, user_constraint_status_objects)
        accept_resp = AcceptHardCodedBasedResponse(hard_coded_responses)
        reject_resp = RejectHardCodedBasedResponse(hard_coded_responses)

        # Initialize recommender action classifier
        rec_actions = [Answer(answer_resp),
                       Recommend(user_constraint_status_objects, hard_coded_responses, recc_resp),
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
        self.init_msg = 'Hello I am your conversational recommender! Please state your preference!'
        for hard_coded_response in hard_coded_responses:
            if hard_coded_response['action'] == 'InitMessage':
                self.init_msg = hard_coded_response['response']

    def run(self) -> None:
        """
        Run the conv rec system. User can quit by typing 'quit' or 'q'.
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

        :param user_input: input from the user
        """
        self.is_gpt_retry_notified = False
        self.is_warning_notified = False
        return self.dialogue_manager.get_response(user_input)

    def notify_warning(self) -> None:
        """
        Notify this object about warnings.
        """
        if not self.is_warning_notified:
            self.user_interface.display_warning(
                "Sorry.. running into some difficulties, this is going to take longer than usual.")
        self.is_warning_notified = True
