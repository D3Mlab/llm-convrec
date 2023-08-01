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
    :param llm_wrapper: object to make request to LLM
    :param hard_coded_responses: list that defines all hard coded responses
    """

    state_manager: StateManager
    _user_intents_classifier: UserIntentsClassifier
    _rec_actions_classifier: RecActionsClassifier
    _default_response: str
    _llm_wrapper: LLMWrapper
    _hard_coded_responses: list[dict]

    def __init__(self, state_manager: StateManager, user_intents_classifier: UserIntentsClassifier,
                 rec_actions_classifier: RecActionsClassifier, llm_wrapper: LLMWrapper, hard_coded_responses: list[dict]):
        self.state_manager = state_manager
        self._user_intents_classifier = user_intents_classifier
        self._rec_actions_classifier = rec_actions_classifier
        self._hard_coded_responses = hard_coded_responses
        self._llm_wrapper = llm_wrapper

    def get_response(self, user_input: str) -> str:
        """
        Generate and return recommender's response by classifying user intent, update state,
        classify recommender action, and generating the response.

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
            # If not classified into any user intent, recommender give default response.
            rec_response = ""
            for response_dict in self._hard_coded_responses:
                if response_dict['action'] == 'DefaultResponse':
                    rec_response = response_dict['response']
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

    def _generate_response(self, rec_actions: list[RecAction]) -> str:
        """
        Helper function to generate recommender's response.

        :param rec_actions: list of recommender actions
        :return: response from the recommender
        """
        # Note only works for 1 rec action for now
        for action in rec_actions:
            resp = action.get_response(self.state_manager)            
            self.state_manager.store_response(resp)
            return resp
