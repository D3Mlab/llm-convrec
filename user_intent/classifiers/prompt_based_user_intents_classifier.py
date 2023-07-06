from textwrap import dedent

from intelligence.llm_wrapper import LLMWrapper
from state.state_manager import StateManager
from user_intent.user_intent import UserIntent
from user_intent.classifiers.user_intents_classifier import UserIntentsClassifier


class PromptBasedUserIntentsClassifier(UserIntentsClassifier):
    """
    Implementation of UserIntentClassifier that classifies the user intent by generating prompt for LLM.

    :param user_intents: all possible user intents
    :param llm_wrapper: LLMWrapper used for classifying the user intent
    """
    _user_intents: list[UserIntent]
    _llm_wrapper: LLMWrapper

    def __init__(self, user_intents: list[UserIntent], llm_wrapper: LLMWrapper):
        super().__init__(user_intents)
        self._user_intents = user_intents
        self._llm_wrapper = llm_wrapper

    def classify(self, curr_state: StateManager) -> list[UserIntent]:
        """
        Returns one of the user intent in self._user_intents corresponding to the most recent user's input in the
        given state.
        Returns None if none of user intents in self._user_intents corresponds to the most recent user's input.

        :param curr_state: current state representing the conversation
        :return: user intent corresponding to the most recent user's input or None if there are no user intent
                 corresponding to the user's input
        """
        user_intent_name: str
         
        if len(curr_state.get("conv_history")) == 0:
            return []

        # generate prompt and call LLM to classify user intent
        prompt = self._generate_prompt(curr_state)
        
        user_intent_name = self._llm_wrapper.make_request(prompt)
        user_intent_name = user_intent_name.removesuffix(".")
        user_intent_name = user_intent_name.strip()
        
        # TODO: if the prompt returns a user intent that is not exactly like we expect still return the user intent (do some string stuff) 
        user_intents = []
        for user_intent in self._user_intents:
            if user_intent.get_name() == user_intent_name:
                user_intents.append(user_intent)
        
        return user_intents

    def _generate_prompt(self, curr_state: StateManager) -> str:
        """
        Generate and return prompt used for user intent classification based on the current state.

        :param curr_state: current state representing the conversation
        :return: prompt to LLM used for user intent classification
        """
        current_user_input = ""
        previous_recommender_response = ""
        previous_user_input = ""

        if len(curr_state.get("conv_history")) >= 1:
            current_user_input = curr_state.get("conv_history")[-1].get_content()

        if len(curr_state.get("conv_history")) >= 2:
            previous_recommender_response = curr_state.get("conv_history")[-2].get_content()

        if len(curr_state.get("conv_history")) >= 3:
            previous_user_input = curr_state.get("conv_history")[-3].get_content()

        return dedent(f"""
        Classify the user’s intent from the current user’s input into categories listed below, given the previous conversation between recommender and user. Only indicate the category name and nothing else. 

        {self._generate_user_intents_description()}

        User’s previous input: \"{previous_user_input}\"

        Recommender’s previous response: \"{previous_recommender_response}\"

        User’s current input: \"{current_user_input}\"""")

    def _generate_user_intents_description(self):
        """
        Generate descriptions of the user intents in self._user_intents in the following format:
            Categories with description:
            <user intent 1>: <description of the user intent 1>
            <user intent 2>: <description of the user intent 2>
            ...

        :return: descriptions of the user intents in self._user_intents
        """
        result = "Categories with description: \n"
        for user_intent in self._user_intents:
            result += f"{user_intent.get_name()}: {user_intent.get_description()}\n"
        return result
