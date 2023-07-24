import re
from intelligence.llm_wrapper import LLMWrapper
from state.state_manager import StateManager
from user_intent.user_intent import UserIntent
from user_intent.classifiers.user_intents_classifier import UserIntentsClassifier
from user_intent.provide_preference import ProvidePreference
import threading
from utility.thread_utility import start_thread


class MultilabelUserIntentsClassifier(UserIntentsClassifier):

    def __init__(self, user_intents: list[UserIntent], llm_wrapper: LLMWrapper, config: dict, force_provide_preference: bool = False):
        super().__init__(user_intents)
        self._user_intents = user_intents.copy()
        self._llm_wrapper = llm_wrapper
        self._provide_preference = None
        
        self.enable_threading = config['ENABLE_MULTITHREADING']
            
        if force_provide_preference:
            for user_intent in self._user_intents:
                if isinstance(user_intent, ProvidePreference):
                    self._provide_preference = user_intent
                    self._user_intents.remove(user_intent)
                    break
    
    '''def classify(self, curr_state: StateManager) -> list[UserIntent]:
        """
        Returns list of user intents identified in order of decreasing priority, after getting a score for each user intent possible

        :param curr_state: current state representing the conversation
        :return: (first element) in list of user intents identified in order of decreasing priority
        """
        score_list = []
        for user_intent in self._user_intents:
            prompt = user_intent.get_prompt_for_score(curr_state)
            str = self._llm_wrapper.make_request(prompt)
            numbers = re.findall(r'\b\d+\b', str)
            if numbers:
                score = float(numbers[0])
            else:
                score = 0
            score_list.append((user_intent, score))
        if self._provide_preference is not None:
            score_list.append((self._provide_preference, 10))
        sorted_user_intents = sorted(score_list, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_user_intents if x[1] >= 5]'''
    
    def classify(self, curr_state: StateManager) -> list[UserIntent]:
        """
        Returns list of user intents identified, after getting a Boolean result for each user intent possible.

        :param curr_state: current state representing the conversation
        :return: list of user intents identified
        """
        intent_list = []
        thread_list = []
        for user_intent in self._user_intents:
            if (self.enable_threading):
                thread = threading.Thread(
                    target=self._classify_one_intent, args=(user_intent, curr_state, intent_list))
                thread_list.append(thread)
            else:
                self._classify_one_intent(user_intent, curr_state, intent_list)
                
        if (self.enable_threading):
            start_thread(thread_list)
        
        if self._provide_preference is not None:
            intent_list.append(self._provide_preference)
        
        return intent_list

    def _classify_one_intent(self, user_intent: UserIntent, curr_state: StateManager, intent_list: list[UserIntent]):
        """
        Returns either true/false indicating if the user input should be classified into the user intent.
        :param curr_state: current state representing the conversation
        :param user_intent: the user intent that you are trying to see if the user input should be classified into
        :param intent_list: list of user intents that input is classified into
        :return: none
        """
        prompt = user_intent.get_prompt_for_classification(curr_state)
        str = self._llm_wrapper.make_request(prompt)
        if "True" in str:
            intent_list.append(user_intent)