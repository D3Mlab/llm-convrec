from rec_action.rec_actions_classifier import RecActionsClassifier
from state.state_manager import StateManager
from user_intent.classifiers.user_intents_classifier import UserIntentsClassifier
from rec_action.rec_action import RecAction
from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
import logging

logger = logging.getLogger('dialogue_manager')


class DialogueManager:
    """
    Class responsible for providing recommender's response by classifying user intent, update state,
    classify recommender action, and generating the response.

    :param state_manager: object to keep track of the current state of the conversation
    :param user_intents_classifier: object used to classify user intent
    :param rec_actions_classifier: object used to classify recommender actions
    :param default_response: default response provided when appropriate response cannot be determined
    :param llm_wrapper: object to make request to LLM
    """

    state_manager: StateManager
    _user_intents_classifier: UserIntentsClassifier
    _rec_actions_classifier: RecActionsClassifier
    _default_response: str
    _llm_wrapper: LLMWrapper

    def __init__(self, state_manager: StateManager, user_intents_classifier: UserIntentsClassifier,
                 rec_actions_classifier: RecActionsClassifier, llm_wrapper: LLMWrapper, default_response="Could you provide more information?"):
        self.state_manager = state_manager
        self._user_intents_classifier = user_intents_classifier
        self._rec_actions_classifier = rec_actions_classifier
        self._default_response = default_response
        self._llm_wrapper = llm_wrapper

    def get_response(self, user_input: str) -> str:
        """
        Generate and return recommender's response by classifying user intent, update state,
        classify recommender action, and generating the response.
        Return self._default_response if any other appropriate response cannot be determined.

        :param user_input: current user's input
        :return: response from the recommender
        """
        logger.debug(f'user_input="{user_input}"')

        message = Message("user", user_input)
        self.state_manager.update_conv_history(message)

        user_intents = self._user_intents_classifier.classify(
            self.state_manager)
        logger.debug(f'user_intents={str(user_intents)}')
        if not user_intents:
            rec_response = self._default_response
            self.state_manager.store_response(rec_response)
            logger.warning(
                f"User input, \"{user_input}\" was not classified to any of the user intent.")
        else:
            self.state_manager.store_user_intents(user_intents)
            rec_actions = self._rec_actions_classifier.classify(
                self.state_manager)

            logger.debug(f'rec_actions={str(rec_actions)}')
            if not rec_actions:
                rec_response = self._default_response
                self.state_manager.store_response(rec_response)
                logger.warning(
                    f"User input, \"{user_input}\" was not classified to any of the recommender action.")
            else:
                self.state_manager.store_rec_actions(rec_actions)
                rec_response = self._generate_response(rec_actions)

        logger.debug(f'rec_response="{rec_response}"')
        logger.debug(f"state_manager={str(self.state_manager)}")
        return rec_response

    def _generate_response(self, rec_actions: list[RecAction]):
        """
        Helper function to generate recommender's response.

        :param rec_actions: list of recommender actions
        :return: response from the recommender
        """

        hard_coded_llm_resp = "I couldn't find any relevant information in the product database to help me respond. Based on my internal knowledge, which does not include any information after 2021..." + '\n'

        # Note only 1 action in MVP
        for action in rec_actions:
            hard_coded_resp = "  "
            if action.is_response_hard_coded():
                hard_coded_resp = action.get_hard_coded_response(
                    self.state_manager)

            # If could not create a hard coded response
            if not action.is_response_hard_coded():
                rec_response = action.get_prompt_response(self.state_manager)
                self.state_manager.store_response(rec_response)
                return rec_response

            else:
                self.state_manager.store_response(hard_coded_resp)
                return hard_coded_resp

