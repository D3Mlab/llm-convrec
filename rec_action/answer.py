import logging
from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.inquire import Inquire
from state.message import Message
from decimal import Decimal
from string import ascii_letters
from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.filter.filter_applier import FilterApplier
from information_retrievers.information_retrieval import InformationRetrieval
from intelligence.llm_wrapper import LLMWrapper
from domain_specific_config_loader import DomainSpecificConfigLoader
from jinja2 import Environment, FileSystemLoader
import yaml
from warning_observer import WarningObserver

logger = logging.getLogger('answer')


class Answer(RecAction):
    """
    Class representing Answer recommender action.
    :param config: config values from system_config.yaml
    :param priority_score_range: range of scores for smth TODO: fill in smth
    :param information_retriever: information retriever that is used to fetch restaurant recommendations
    :param llm_wrapper: object to make request to LLM
    """

    def __init__(self, config: dict, llm_wrapper: LLMWrapper, filter_items: FilterApplier,
                 information_retriever: InformationRetrieval, domain: str,
                 observers=None,
                 priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)
        self._filter_items = filter_items
        self._domain = domain
        self._observers = observers

        if config["NUM_REVIEWS_TO_RETURN"]:
            self._num_of_reviews_to_return = int(
                config["NUM_REVIEWS_TO_RETURN"])

        if self.is_response_hard_coded():
            self._information_retriever = information_retriever
            self._llm_wrapper = llm_wrapper
        else:
            raise Exception("Need to change get_prompt()!")

        self._prompt = ""

        with open("system_config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        env = Environment(loader=FileSystemLoader(
            self.config['ANSWER_PROMPTS_PATH']), trim_blocks=True, lstrip_blocks=True)

        self.gpt_template = env.get_template(
            self.config['ANSWER_GPT_PROMPT'])

        self.mult_qs_template = env.get_template(
            self.config['ANSWER_MULT_QS_PROMPT'])

        self.verify_metadata_template = env.get_template(
            self.config['ANSWER_VERIFY_METADATA_RESP_PROMPT'])

        self.format_mult_qs_template = env.get_template(
            self.config['ANSWER_MULT_QS_FORMAT_RESP_PROMPT'])

        self.format_mult_resp_template = env.get_template(
            self.config['ANSWER_FORMAT_MULTIPLE_RESP_PROMPT'])

        self.extract_category_template = env.get_template(
            self.config['ANSWER_EXTRACT_CATEGORY_PROMPT'])

        self.metadata_template = env.get_template(
            self.config['ANSWER_ATTR_PROMPT'])

        self.ir_template = env.get_template(
            self.config['ANSWER_IR_PROMPT'])

        domain_specific_config_loader = DomainSpecificConfigLoader()

        self._extract_category_few_shots \
            = domain_specific_config_loader.load_answer_extract_category_fewshots()

        self._ir_prompt_few_shots \
            = domain_specific_config_loader.load_answer_ir_fewshots()

        self._separate_qs_prompt_few_shots \
            = domain_specific_config_loader.load_answer_separate_questions_fewshots()

        self._verify_metadata_prompt_few_shots \
            = domain_specific_config_loader.load_answer_verify_metadata_resp_fewshots()

    def get_name(self) -> str:
        """
        Returns the name of this recommender action.

        :return: name of this recommender action
        """
        return "Answer"

    def get_description(self) -> str:
        """
        Returns the description of this recommender action.

        :return: description of this recommender action
        """
        return "Recommender answers the question issued by the user"

    def get_priority_score(self, state_manager: StateManager) -> float:
        """
        Returns the score representing how much this is appropriate recommender action for the current conversation.

        :param state_manager: current state representing the conversation
        :return: score representing how much this is appropriate recommender action for the current conversation.
        """
        if state_manager.get("unsatisfied_goals") is not None:
            for goal in state_manager.get("unsatisfied_goals"):
                if isinstance(goal["user_intent"], Inquire):
                    return self.priority_score_range[0] + goal["utterance_index"] / len(state_manager.get("conv_history")) * (self.priority_score_range[1] - self.priority_score_range[0])
        return self.priority_score_range[0] - 1

    def get_prompt_response(self, state_manager: StateManager) -> str | None:
        """
        Return prompt based recommender's response corresponding to this action.

        :param state_manager: current state representing the conversation
        :return: prompt based recommender's response corresponding to this action
        """
        return None

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """

        curr_mentioned_restaurants: list[RecommendedItem] = state_manager.get(
            "curr_items")

        answers = {}

        if curr_mentioned_restaurants is not None:

            user_questions = self._seperate_input_into_multiple_qs(
                state_manager)

            for question in user_questions:
                logger.debug(
                    f'The question is {question}')

                llm_resp = ""
                answers[question] = {}

                for curr_mentioned_restaurant in curr_mentioned_restaurants:
                    logger.debug(
                        f'The recommended restaurant is {curr_mentioned_restaurant.get_name()}')
                    category = self._extract_category_from_input(
                        question, curr_mentioned_restaurant)

                    logger.debug(f'The category is {category}')

                    logger.debug(
                        f'Is the category the LLM classified the user input into valid: {self._is_category_valid(category, curr_mentioned_restaurant)}')

                    if self._is_category_valid(category, curr_mentioned_restaurant):
                        logger.debug("Metadata question!")

                        metadata_resp = (self._create_resp_from_metadata(
                            question, category, curr_mentioned_restaurant))

                        answers[question][curr_mentioned_restaurant.get_name()] = metadata_resp

                    if not self._is_category_valid(category, curr_mentioned_restaurant) or metadata_resp == "" or not self._verify_metadata_resp(question, metadata_resp):

                        logger.debug("Non metadata question!")

                        ir_resp = self._create_resp_from_ir(
                            question, curr_mentioned_restaurant)

                        answers[question][curr_mentioned_restaurant.get_name()] = ir_resp

                        if "I do not know" in ir_resp:
                            logger.debug(
                                f'Answer with GPT')

                            prompt = self.gpt_template.render(
                                curr_mentioned_item=curr_mentioned_restaurant, question=question)

                            llm_resp = self._llm_wrapper.make_request(
                                prompt)

                            answers[question][curr_mentioned_restaurant.get_name()] = llm_resp
                    

                mult_rest_resp = self._format_multiple_restaurant_resp(
                    question, curr_mentioned_restaurants, answers[question])
                
                answers[question] = mult_rest_resp

        else:
            return "Please ask questions about previously recommended restaurants."

        return self._format_multiple_qs_resp(state_manager, answers, llm_resp != "")

    def _seperate_input_into_multiple_qs(self, state_manager: StateManager) -> list[str]:
        """
        Takes the users input and splits it into a list of questions
        :param state_manager: the question extracted from the users input
        :returns: list of questions.
        """

        current_user_input = state_manager.get(
            "conv_history")[-1].get_content()

        prompt = self.mult_qs_template.render(
            current_user_input=current_user_input, few_shots=self._separate_qs_prompt_few_shots)

        resp = self._llm_wrapper.make_request(prompt)

        if '\\n' in resp:
            return resp.split('\\n')

        return resp.split('\n')

    def _create_resp_from_ir(self, question: str, curr_mentioned_item: RecommendedItem):
        """
        Returns the string to be returned to the user when using information retrieval
        :param question: the question extracted from the users input
        :param curr_mentioned_item: one of the recommended item user is currently referring to
        :returns: resp to user.
        """
        query = self.convert_state_to_query(
            question, curr_mentioned_item)

        logger.debug(f'Query: {query}')

        item_ids_to_keep = self._filter_items.filter_by_current_item([curr_mentioned_item])
        reviews = self._information_retriever.get_best_matching_reviews_of_item(
            query, self._num_of_reviews_to_return, item_ids_to_keep)[0]

        return self._format_review_resp(
            question, reviews, curr_mentioned_item)

    def _is_category_valid(self, classified_category: str, recommended_restaurant: RecommendedItem) -> bool:
        """
        Returns a bool representing if the classified category is valid.
        A cateogry is valid if it is classified into one of the categories the LLM is told to do.
        :param classified_category: cateogry the LLM classified the user input into
        :param recommended_restaurant: object representing the reccommended restaurant user is referring to
        :return: bool
        """
        valid_categories = []

        for key in recommended_restaurant.get_data():
            valid_categories.append(key)        
            
        for valid_category in valid_categories:
            if self._remove_punct_string(valid_category) in self._remove_punct_string(classified_category):
                return True

        return False

    def _verify_metadata_resp(self, question: str, resp: str) -> bool:
        """
        Sees if the metadata response makes sense given the users question.

        :param resp: string representing the metadata response
        :param question: the question extracted from the users input
        :returns: bool
        """

        prompt = self.verify_metadata_template.render(
            question=question, resp=resp, few_shots=self._verify_metadata_prompt_few_shots)

        valid_resp = self._llm_wrapper.make_request(prompt)

        if 'yes' in self._remove_punct_string(valid_resp):
            return True
        else:
            return False

    def _format_multiple_qs_resp(self, state_manager: StateManager, all_answers: dict, is_llm_res: bool) -> str:
        """
        Returns the response. Returns either an empty string indicating that more work needs to be done to formulate the response or the actual string response.

        :state_manager: current state representing the conversation
        :param all_answers: list of answers to the users question(s)
        :param is_llm_res: boolean indicating if the answer was created using an LLM or not
        :return: str
        """
        
        resp = ""

        if (len(all_answers) > 1):
            user_input = state_manager.get("conv_history")[-1].get_content()

            prompt = self.format_mult_qs_template.render(
                user_input=user_input, all_answers=list(all_answers.values()))
            
            resp = self._llm_wrapper.make_request(prompt)

        else:
            resp = list(all_answers.values())[0]
        
        if is_llm_res:
            resp = "I couldn't find any relevant information in the product database to help me respond. Based on my internal knowledge, which does not include any information after 2021..." + '\n' + resp

        if '"' in resp:
            # get rid of double quotes (gpt sometimes outputs it)
            resp = resp.replace('"', "")
        
        resp = resp.removeprefix('Response to user:').removeprefix('response to user:').strip()

        return resp

    def _format_multiple_restaurant_resp(self, question: str, current_mentioned_restaurants: list[RecommendedItem], answers: dict) -> str:
        """
        Returns the response for one question

        :state_manager: current state representing the conversation
        :param question: the question extracted from the users input
        :param current_restaurants: list of current restaurants user is referring to
        :param answers: dict of answers to the users question where the key is the restaurant and the value is the answer to the question
        :return: str
        """
        resp = ""

        curr_ment_res_names = [current_mentioned_restaurant.get_name()
                               for current_mentioned_restaurant in current_mentioned_restaurants]

        curr_ment_res_names_str = ", ".join(curr_ment_res_names)

        res_to_answ = {
            ', '.join([f'{key}: {val}' for key, val in answers.items()])}

        if (len(answers) > 1):

            prompt = self.format_mult_resp_template.render(
                question=question, curr_ment_item_names_str=curr_ment_res_names_str,
                res_to_answ=res_to_answ, domain=self._domain)

            resp = self._llm_wrapper.make_request(prompt)

        else:
            resp = list(answers.values())[0]

        if '"' in resp:
            # get rid of double quotes (gpt sometimes outputs it)
            resp = resp.replace('"', "")

        return resp

    def _remove_punct_string(self, expr) -> str:
        """ 
        Removes the punctuation, spacing and capitalization from the input. Used to compare strings

        :param expr: string that you want make changes to
        """
        return ''.join([letter.lower() for letter in expr if letter in ascii_letters])

    def _extract_category_from_input(self, question: str, curr_restaurant: RecommendedItem):
        """
        Returns the recommended restaurant object from the users input.
        :param question: the question extracted from the users input
        :param curr_restaurant: object representing the reccommended restaurant user is referring to 
        :return: RecommendedRestaurant.
        """

        categories = ""

        for key in curr_restaurant.get_data():
            categories += f" {key},"

        categories += " or none"

        prompt = self.extract_category_template.render(
            curr_item=curr_restaurant, categories=categories, question=question, domain=self._domain,
            few_shots=self._extract_category_few_shots)

        return self._llm_wrapper.make_request(prompt)

    def _create_resp_from_metadata(self, question: str, category: str, recommended_restaurant: RecommendedItem) -> str:
        """
        Returns the string to be returned to the user using metadata
        :param question: the question extracted from the users input
        :param category: category of metadata that information required to answer the question is stored
        :param recommended_restaurant: object representing the reccommended restaurant user is referring to 
        :return: resp to user.
        """

        resp = ""

        for key, val in recommended_restaurant.get_data().items():
            if self._remove_punct_string(key) in self._remove_punct_string(category):

                prompt = self.metadata_template.render(
                    question=question, key=key, val=val)
                
                return self._llm_wrapper.make_request(prompt)

        return resp

    def convert_state_to_query(self, question: str, recommended_restaurant: RecommendedItem) -> str:
        """
        Returns the string to be returned to be used as the query for information retrieval
        :param question: the question extracted from the users input
        :param recommended_restaurant: object representing the reccommended restaurant user is referring to 
        :return: string.
        """

        return question

    def _format_review_resp(self, question: str, reviews: list[str], curr_restaurant: RecommendedItem) -> str:
        """
        Returns the string to be returned to the user from the reviews
        :param question: the question extracted from the users input
        :param reviews: the list of reviews corresponding to user input
        :return: string.
        """

        try:
            prompt = self.ir_template.render(
                curr_item=curr_restaurant, question=question, reviews=reviews, domain=self._domain,
                few_shots=self._ir_prompt_few_shots)

            resp = self._llm_wrapper.make_request(prompt)
        except:
            # this is very slow
            self._notify_observers()

            logger.debug("Reviews are too long, summarizing...")

            summarized_reviews = []
            for review in reviews:
                summarize_review_prompt = f"""Using a few concise sentences, summarize the following review about a restaurant: {review}"""
                summarized_review = self._llm_wrapper.make_request(
                    summarize_review_prompt)
                summarized_reviews.append(summarized_review)

            prompt = self.ir_template.render(
                curr_item=curr_restaurant, question=question, reviews=summarized_reviews, domain=self._domain,
                few_shots=self._ir_prompt_few_shots)

            return self._llm_wrapper.make_request(prompt)

        return resp

    def _notify_observers(self) -> None:
        """
        Notify observers that there are some difficulties.
        """
        for observer in self._observers:
            observer.notify_warning()

    def is_response_hard_coded(self) -> bool:
        """
        Returns whether hard coded response exists or not.
        :return: whether hard coded response exists or not.
        """
        return True

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