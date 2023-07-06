from textwrap import dedent

from intelligence.llm_wrapper import LLMWrapper
from state.message import Message
from information_retrievers.recommended_item import RecommendedItem

from string import punctuation


class CurrentItemsExtractor:
    """
    Class used to extract the current restaurant the user is referring to from the most recent user's input.

    :param llm_wrapper: LLMWrapper used to extract restaurants
    """
    _llm_wrapper: LLMWrapper

    def __init__(self, llm_wrapper: LLMWrapper) -> None:
        self._llm_wrapper = llm_wrapper

    def extract(self, recommended_restaurants: list[list[RecommendedItem]], conv_history: list[Message]) -> RecommendedItem | None:
        """
        Extract the current restaurant from the most recent user's input in the conv_history and
        return it.

        :param recommended_restaurants: current recommended restaurants as a list of lists where each sublist are recommendations made in one turn.
        :param conv_history: current conversation history
        :return current restaurant

        """
        prompt = self._generate_restaurants_update_prompt(
            conv_history, recommended_restaurants)

        llm_response = self._llm_wrapper.make_request(prompt)
        llm_response = self._clean_string(llm_response)
        curr_mentioned_restaurants = self._get_objects_from_llm_response(
            recommended_restaurants, llm_response)

        return curr_mentioned_restaurants

    def _generate_restaurants_update_prompt(self, conv_history: list[Message], recommended_restaurants: list[list[RecommendedItem]]) -> str:
        """
        Generate and return prompt for extracting current restaurants the user is referring to from the most recent user's input in the
        conversation history.

        :param recommended_restaurants: recommended restaurants as a dictionary
        :param conv_history: current conversation history
        :return prompt for extracting current restaurant from the most recent user's input in the conversation history.
        """

        recc_res_names = [current_mentioned_restaurant.get("name"
        ) for recommeded_restaurants_per_uttr in recommended_restaurants for current_mentioned_restaurant in recommeded_restaurants_per_uttr]

        curr_ment_res_names_str = ", ".join(recc_res_names)

        current_user_input = conv_history[-1].get_content()

        return dedent(f"""
        Extract the restaurant names from the user input and respond only with the names of the restaurant separated by commas and ending with a period. 
        If you cannot extract a restaurant name respond with "None.".

        The user may use short forms or 'slang' when referring to the following restaurants: {curr_ment_res_names_str}. 
        If this is the case then respond with the entire restaurant name, not the slang or short form the user referred to.

        User Input: I don't like thai express, starbucks or subway.
        Response: Thai Express, Starbucks, Subway.

        User Input: I like that they have a patio.
        Response: None.

        User Input: Does thai express have a balcony?
        Response: Thai Express.

        User Input: where is timmy's?
        Response: Tim Hortons.

        User Input: "{current_user_input}".
        Response:

        """)

    def _clean_string(self, llm_response) -> str:
        """
        Get rid of punctuation and leading and trailing spaces
        :param llm_response: response from LLM
        :return string
        """

        return llm_response.strip(punctuation + " ")

    def _get_objects_from_llm_response(self, recommended_restaurants: list[list[RecommendedItem]], llm_response: str) -> RecommendedItem:
        """
        Verify that the response is one of the recommended restaurants and return the object

        :param recommended_restaurants: current recommended restaurants as a list of lists
        :param llm_response: response from LLM
        :return RecommendedRestaurant

        """
        restaurants = []

        if llm_response == 'None':
            return restaurants

        llm_resp_restaurants = llm_response.split(',')

        for recommended_restaurants_per_turn in recommended_restaurants:
            for recommended_restaurant in recommended_restaurants_per_turn:
                if recommended_restaurant.get("name") in llm_resp_restaurants:
                    restaurants.append(recommended_restaurant)

        return restaurants
