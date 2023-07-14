from typing import Dict, Any

import yaml
from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.ask_for_recommendation import AskForRecommendation
from information_retrievers.neural_information_retriever import InformationRetriever
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from state.message import Message
from intelligence.llm_wrapper import LLMWrapper
import logging
import jinja2
from jinja2 import Environment, FileSystemLoader
from state.status import Status

logger = logging.getLogger('recommend')


class Recommend(RecAction):
    """
    Class representing Recommend recommender action.

    :param llm_wrapper: object to make request to LLM
    :param mandatory_constraints: constraints that state must have in order to recommend
    :param information_retriever: information retriever that is used to fetch restaurant recommendations
    :param priority_score_range: range of scores for classifying recaction
    """

    _llm_wrapper: LLMWrapper
    _mandatory_constraints: list[list[str]]
    _information_retriever: InformationRetriever
    _topk_restautants: int
    _topk_reviews: int
    _current_recommended_restaurants: list[RecommendedItem]
    _filter_restaurants: FilterRestaurants
    _env: Environment
    _convert_state_to_query_prompt: jinja2.Template
    _explain_recommendation_prompt: jinja2.Template
    _format_recommendation_prompt: jinja2.Template
    _no_matching_restaurant_prompt: jinja2.Template
    _summarize_review_prompt: jinja2.Template
    _constraint_statuses: list[Status]

    def __init__(self, llm_wrapper: LLMWrapper, filter_restaurants: FilterRestaurants,
                 information_retriever: InformationRetriever, domain: str,constraint_statuses: list, config: dict,
                 constraints_categories: list,
                 priority_score_range=(1, 10)):
        super().__init__(priority_score_range)
       
        self._mandatory_constraints = [constraint_category['key'] for constraint_category in
                                             constraints_categories if constraint_category['is_mandatory']]
        
        print(self._mandatory_constraints)
        
        self._filter_restaurants = filter_restaurants
        self._information_retriever = information_retriever
        self._current_recommended_restaurants = []
        self._llm_wrapper = llm_wrapper
        self._domain = domain
        self._constraint_statuses = constraint_statuses


        if config["TOPK_RESTAURANTS"] and config["TOPK_REVIEWS"] and config["RECOMMEND_PROMPTS_PATH"] \
                and config["CONVERT_STATE_TO_QUERY_PROMPT_FILENAME"] \
                and config["EXPLAIN_RECOMMENDATION_PROMPT_FILENAME"] and config["FORMAT_RECOMMENDATION_PROMPT_FILENAME"]\
                and config["NO_MATCHING_RESTAURANT_PROMPT_FILENAME"] and config["SUMMARIZE_REVIEW_PROMPT_FILENAME"]:
            self._topk_restautants = int(config["TOPK_RESTAURANTS"])
            self._topk_reviews = int(config["TOPK_REVIEWS"])
            
            self._env = Environment(loader=FileSystemLoader(
                config['RECOMMEND_PROMPTS_PATH']))
            self._convert_state_to_query_prompt = self._env.get_template(
                config['CONVERT_STATE_TO_QUERY_PROMPT_FILENAME'])
            self._explain_recommendation_prompt = self._env.get_template(
                config['EXPLAIN_RECOMMENDATION_PROMPT_FILENAME'])
            self._format_recommendation_prompt = self._env.get_template(
                config['FORMAT_RECOMMENDATION_PROMPT_FILENAME'])
            self._no_matching_restaurant_prompt = self._env.get_template(
                config['NO_MATCHING_RESTAURANT_PROMPT_FILENAME'])
            self._summarize_review_prompt = self._env.get_template(
                config['SUMMARIZE_REVIEW_PROMPT_FILENAME'])

        else:
            raise Exception("A config parameter is missing!")

    def get_name(self):
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Recommend"

    def get_description(self):
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "The recommender provides a recommendation either by directly presenting the item or asking the " \
               "user if she knows of the place."

    def get_response_info(self, state_manager: StateManager) -> dict:
        """
        Returns recommender's response corresponding to this recommender action based on the given state.
        It asks LLM to generate recommendation based on the current constraints.

        :param state_manager: current state representing the conversation
        :return: recommender's response corresponding to this recommender action based on the current state.
        """
        return {"prompt": self.get_prompt(state_manager)}

    def get_prompt(self, state_manager: StateManager) -> str | None:
        """
        Return prompt that can be inputted to LLM to produce recommender's response.
        Return None if it doesn't exist.

        :param state_manager: current state representing the conversation
        :return: prompt that can be inputted to LLM to produce recommender's response or None if it doesn't exist.
        """
        query = self.convert_state_to_query(state_manager)

        logger.debug(f'Query: {query}')

        filtered_embedding_matrix = \
            self._filter_restaurants.filter_by_constraints(state_manager)

        try:
            self._current_recommended_restaurants = \
                self._information_retriever.get_best_matching_items(query, self._topk_restautants,
                                                                    self._topk_reviews, filtered_embedding_matrix)
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            loc_name = self._get_all_loc_name_in_text(state_manager)
            return self._get_prompt_for_no_matching_item(loc_name)

        explanation = self._get_explanation_for_each_restaurant(state_manager)
        return self._get_prompt_to_format_recommendation(explanation)

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """
        query = self.convert_state_to_query(state_manager)

        logger.debug(f'Query: {query}')

        filtered_embedding_matrix = \
            self._filter_restaurants.filter_by_constraints(state_manager)

        try:
            self._current_recommended_restaurants = \
                self._information_retriever.get_best_matching_items(query, self._topk_restautants,
                                                                    self._topk_reviews, filtered_embedding_matrix)
            return self._format_hard_coded_resp(self._current_recommended_restaurants)
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            loc_name = self._get_all_loc_name_in_text(state_manager)
            return "Sorry, there is no restaurant near " + loc_name + "."

    def _get_prompt_for_no_matching_item(self):
        """
        Get the prompt to get recommendation text with explanation.
        :param loc_name: location name(s)
        :return: prompt to use when there is no matching item
        """
        return self._no_matching_restaurant_prompt.render(domain=self._domain)

    def _get_all_loc_name_in_text(self, state_manager: StateManager):
        """
        Get all locations in hard_constraints in a plain text.
        :param state_manager: current state_manager
        :return: all locations name in a plain text
        """
        locations = state_manager.get("hard_constraints").get("location")
        return 'or '.join([f'{val}' for val in locations])

    def _get_prompt_to_format_recommendation(self, explanation: dict):
        """
        Get the prompt to get recommendation text with explanation.
        :param explanation: explanation of why the restaurants are recommended
        :return: prompt to get recommendation text with explanation
        """
        item_names = ' and '.join(
            [f'{rec_restaurant.get("name")}' for rec_restaurant in self._current_recommended_restaurants])
        explanation_str = ', '.join(
            [f'{key}: {val}' for key, val in explanation.items()])
        return self._format_recommendation_prompt.render(
            item_names=item_names, explanation=explanation_str, domain=self._domain)

    def _get_explanation_for_each_restaurant(self, state_manager: StateManager) -> dict[Any, str]:
        """
        Returns the explanation on why recommending each restaurant
        :param state_manager: current state representing the conversation
        :return: explanation for each restaurant stored in dict where key is restaurant name and value is explanation
        """
        explanation = {}
        for rec_restaurant in self._current_recommended_restaurants:
            restaurant_name = rec_restaurant.get("name")
            hard_constraints = state_manager.get('hard_constraints').copy()
            if "location" in hard_constraints:
                hard_constraints['location'] = []
            data = state_manager.to_dict()
            soft_constraints = {}
            for key, value in data.items():
                if key == "soft_constraints":
                    soft_constraints = value
            metadata = self._get_metadata_of_rec_item(rec_restaurant)
            reviews = rec_restaurant.get_most_relevant_review()
            try:
                prompt = self._get_prompt_to_explain_recommendation(restaurant_name, metadata, reviews,
                                                                    hard_constraints, soft_constraints)

                explanation[restaurant_name] = self._llm_wrapper.make_request(
                    prompt)
            except Exception as e:
                logger.debug(f'There is an error: {e}')
                # this is very slow
                print(
                    'Sorry.. running into some difficulties, this is going to take longer than ususal.')

                logger.debug("Reviews are too long, summarizing...")

                constraints = hard_constraints.copy()
                if soft_constraints is not None:
                    constraints.update(soft_constraints)
                summarized_reviews = []
                for review in rec_restaurant.get_most_relevant_review():
                    summarize_review_prompt = self._get_prompt_to_summarize_review(
                        constraints, review)
                    summarized_review = self._llm_wrapper.make_request(
                        summarize_review_prompt)
                    summarized_reviews.append(summarized_review)

                prompt = self._get_prompt_to_explain_recommendation(restaurant_name, metadata, summarized_reviews,
                                                                    hard_constraints, soft_constraints)

                explanation[restaurant_name] = self._llm_wrapper.make_request(
                    prompt)

        return explanation

    def _get_prompt_to_explain_recommendation(self, item_names: str, metadata: str, reviews: list[str],
                                              hard_constraints: dict, soft_constraints: dict) -> str:
        """
        Get the prompt to get explanation.
        :param item_names: item name
        :param metadata: metadata of the restaurant
        :param reviews: reviews of the restaurant
        :param hard_constraints: hard constraints in current statemanager
        :param soft_constraints: soft constraints in current statemanager
        :return: prompt to get explanation
        """
        return self._explain_recommendation_prompt.render(
            item_names=item_names, metadata=metadata, reviews=reviews,
            hard_constraints=hard_constraints, soft_constraints=soft_constraints, domain=self._domain)

    def _get_prompt_to_summarize_review(self, constraints: dict, review: str) -> str:
        """
        Get prompt to summarize a review.
        :param constraints: both hard and soft constraints combined in current statemanager
        :param review: a review to be summarized
        """
        constraints_str = ', '.join(
            [f'{key}: {val}' for key, val in constraints.items()])
        return self._summarize_review_prompt.render(constraints=constraints_str, review=review, domain=self._domain)

    def _get_metadata_of_rec_item(self, recommended_restaurant: RecommendedItem):
        """
        Get metadata of a restaurant used for recommend
        :param recommended_restaurant: recommended restaurant whose metadata to be returned
        """
        metadata = f"""location: at {recommended_restaurant.get('address')}, """
        attributes = ', '.join(
            [f'{key}: {val}' for key, val in recommended_restaurant.get("attributes").items()])
        metadata += attributes
        return metadata

    def convert_state_to_query(self, state_manager: StateManager) -> str:
        """
        Convert this state to query and return it. 

        :param state_manager: current state_manager
        :return: query used for information retriever
        """
        hard_constraints = state_manager.get('hard_constraints')
        data = state_manager.to_dict()
        soft_constraints = data.get('soft_constraints')
        prompt = self._convert_state_to_query_prompt.render(
            hard_constraints=hard_constraints, soft_constraints=soft_constraints, domain=self._domain)
        query = self._llm_wrapper.make_request(prompt)

        return query

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.
        :return: whether hard coded response exists or not.
        """
        return False

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        hard_constraints = state_manager.get("hard_constraints")
        is_ready = hard_constraints is not None and all(any(hard_constraints.get(key) is not None and
                                                            hard_constraints.get(key) != [] for key in lst)
                                                        for lst in self._mandatory_constraints)
        
        for constraint in self._constraint_statuses:
            if constraint.get_status() is None or constraint.get_status() == 'invalid':
                is_ready = False
                break
        
        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], AskForRecommendation) and is_ready:
                    return self.priority_score_range[0] + goal["utterance_index"] / len(
                        state_manager.get("conv_history")) * (
                        self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def _format_hard_coded_resp(self, recommended_restaurants: list[RecommendedItem]) -> str:
        """
        Formats the hard coded recommender response.
        Returns the generated string from the list of recommended restaurants.

        :param recommended_restaurants: list of recommended restaurants

        :return: response to display to user
        """

        recomm_resp = "How about "
        for restaurant in recommended_restaurants:
            recomm_resp += f'{restaurant.get("name")} at {restaurant.get("address")}, {restaurant.get("city")} with {restaurant.get("stars")} stars out of {restaurant.get("review_count")} reviews or '

        return f'{recomm_resp[:-4]}?'

    def update_state(self, state_manager: StateManager, response: str, **kwargs):
        """
        Updates the state based off of recommenders response

        :param state_manager: current state representing the conversation
        :param response: recommender response msg that is returned to the user
        :param **kwargs: misc. arguments

        :return: none
        """

        message = Message("recommender", response)
        state_manager.update_conv_history(message)

        # Store recommended restaurants
        recommended_restaurants = state_manager.get('recommended_items')

        if recommended_restaurants is None:
            recommended_restaurants = []

        if self._current_recommended_restaurants != []:
            recommended_restaurants.append(
                self._current_recommended_restaurants)

            state_manager.update("recommended_items",
                                 recommended_restaurants)

            # Store new currrent restaurants
            state_manager.update("curr_items",
                                 self._current_recommended_restaurants)

        # TODO: store justification?
