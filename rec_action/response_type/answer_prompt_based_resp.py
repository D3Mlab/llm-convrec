from rec_action.response_type.response import Response
from state.state_manager import StateManager
from information_retriever.item.recommended_item import RecommendedItem
from information_retriever.filter.filter_applier import FilterApplier
from information_retriever.information_retrieval import InformationRetrieval
from intelligence.llm_wrapper import LLMWrapper
from domain_specific_config_loader import DomainSpecificConfigLoader
from utility.thread_utility import start_thread
from warning_observer import WarningObserver

import logging
import threading
from jinja2 import Environment, FileSystemLoader, Template
from string import ascii_letters


logger = logging.getLogger('answer')


class AnswerPromptBasedResponse(Response):
    """
    Class representing the prompt based response for answer

    :param config: config for this system
    :param llm_wrapper: wrapper of LLM used to generate response
    :param filter_applier: object used to apply filter items
    :param information_retriever: object used to retrieve item reviews based on the query
    :param domain: domain of the recommendation (e.g. restaurants)
    :param hard_coded_responses: list that defines every hard coded response
    :param extract_category_few_shots: few shot examples used for extracting category from user's question
    :param ir_prompt_few_shots: few shot examples used to answer question using IR
    :param separate_qs_prompt_few_shots: few shot examples used to separate question in to multiple individual questions
    :param observers: observers that gets notified when reviews must be summarized, so it doesn't exceed
    """

    _num_of_reviews_to_return: int
    _filter_applier: FilterApplier
    _information_retriever: InformationRetrieval
    _llm_wrapper: LLMWrapper
    _prompt: str
    _domain: str
    _observers: list[WarningObserver]
    _hard_coded_responses: list[dict]
    _mult_qs_template: Template
    _format_mult_qs_template: Template
    _format_mult_resp_template: Template
    _extract_category_template: Template
    _metadata_template: Template
    _ir_template: Template
    _enable_threading: bool
    _extract_category_few_shots: list[dict]
    _ir_prompt_few_shots: list[dict]
    _separate_qs_prompt_few_shots: list[dict]

    def __init__(self, config: dict, llm_wrapper: LLMWrapper, filter_applier: FilterApplier,
                 information_retriever: InformationRetrieval, domain: str, hard_coded_responses: list[dict],
                 extract_category_few_shots: list[dict], ir_prompt_few_shots: list[dict],
                 separate_qs_prompt_few_shots: list[dict], observers=None) -> None:
        
        self._filter_applier = filter_applier
        self._domain = domain
        self._observers = observers

        self._information_retriever = information_retriever
        self._llm_wrapper = llm_wrapper

        self._hard_coded_responses = hard_coded_responses

        self._num_of_reviews_to_return = int(
            config["NUM_REVIEWS_TO_RETURN"])

        env = Environment(loader=FileSystemLoader(
            config['ANSWER_PROMPTS_PATH']), trim_blocks=True, lstrip_blocks=True)

        self._mult_qs_template = env.get_template(
            config['ANSWER_MULT_QS_PROMPT'])

        self._format_mult_qs_template = env.get_template(
            config['ANSWER_MULT_QS_FORMAT_RESP_PROMPT'])

        self._format_mult_resp_template = env.get_template(
            config['ANSWER_FORMAT_MULTIPLE_RESP_PROMPT'])

        self._extract_category_template = env.get_template(
            config['ANSWER_EXTRACT_CATEGORY_PROMPT'])

        self._metadata_template = env.get_template(
            config['ANSWER_METADATA_PROMPT'])

        self._ir_template = env.get_template(
            config['ANSWER_IR_PROMPT'])
        
        self._enable_threading = config['ENABLE_MULTITHREADING']

        self._extract_category_few_shots = extract_category_few_shots
        self._ir_prompt_few_shots = ir_prompt_few_shots
        self._separate_qs_prompt_few_shots = separate_qs_prompt_few_shots

    def get(self, state_manager: StateManager) -> str | None:
        """
        Get the response to be returned to user

        :param state_manager: current representation of the state
        :return: response to be returned to user
        """

        curr_mentioned_items: list[RecommendedItem] = state_manager.get(
            "curr_items")

        answers = {}

        if curr_mentioned_items is not None:

            user_questions = self._separate_input_into_multiple_qs(state_manager)
            
            thread_list = []

            for question in user_questions:
                if self._enable_threading:
                    thread_list.append(threading.Thread(
                        target=self._get_resp_one_q, args=(question, curr_mentioned_items, answers)))
                else:
                    self._get_resp_one_q(question, curr_mentioned_items, answers)
                
            if self._enable_threading:
                start_thread(thread_list)
            
        else:
            for response_dict in self._hard_coded_responses:
                if response_dict['action'] == 'NoAnswer':
                    return response_dict['response']

        return self._format_multiple_qs_resp(state_manager, answers)

    def _get_resp_one_q(self, question: str, curr_mentioned_items: list[RecommendedItem], answers: dict) -> None:
        """
        Get the response for one question

        :param question: question extracted from user input
        :param curr_mentioned_items: list of recommended items that user is referring to
        :answers: list of answers to each question / item the user has mentioned
        :return: response to be returned to user
        """
        logger.debug(f'The question is {question}')

        answers[question] = {}

        for curr_mentioned_item in curr_mentioned_items:
            logger.debug(
                f'The recommended {self._domain} is {curr_mentioned_item.get_name()}')
            category = self._extract_category_from_input(
                question, curr_mentioned_item)

            logger.debug(f'The category is {category}')

            logger.debug(
                f'Is the category the LLM classified the user input into valid: {self._is_category_valid(category, curr_mentioned_item)}')

            if self._is_category_valid(category, curr_mentioned_item):
                logger.debug("Metadata question!")
                self.answer_type = "metadata"

                metadata_resp = self._create_resp_from_metadata(
                    question, category, curr_mentioned_item)

                answers[question][curr_mentioned_item.get_name()] = metadata_resp

            if not self._is_category_valid(category, curr_mentioned_item) or "I do not know" in metadata_resp:
                self.answer_type = "ir"

                logger.debug("Non metadata question!")

                ir_resp = self._create_resp_from_ir(
                    question, curr_mentioned_item)

                if "I do not know" in ir_resp:
                    ir_resp = "I don't have access to the information to answer the question."

                answers[question][curr_mentioned_item.get_name()] = ir_resp

        mult_item_resp = self._format_multiple_item_resp(
            question, curr_mentioned_items, answers[question])
        
        answers[question] = mult_item_resp

    def _separate_input_into_multiple_qs(self, state_manager: StateManager) -> list[str]:
        """
        Takes the users input and splits it into a list of questions

        :param state_manager: the question extracted from the users input
        :returns: list of questions.
        """

        current_user_input = state_manager.get(
            "conv_history")[-1].get_content()

        prompt = self._mult_qs_template.render(
            current_user_input=current_user_input, few_shots=self._separate_qs_prompt_few_shots)

        resp = self._llm_wrapper.make_request(prompt)

        if '\\n' in resp:
            return resp.split('\\n')

        return resp.split('\n')

    def _create_resp_from_ir(self, question: str, curr_mentioned_item: RecommendedItem) -> str:
        """
        Returns the string to be returned to the user when using information retrieval

        :param question: the question extracted from the users input
        :param curr_mentioned_item: one of the recommended item user is currently referring to
        :returns: response to user.
        """
        query = self.convert_state_to_query(
            question)

        logger.debug(f'Query: {query}')

        item_index = self._filter_applier.filter_by_current_item(curr_mentioned_item)

        try:
            reviews = self._information_retriever.get_best_matching_reviews_of_item(
                query, self._num_of_reviews_to_return, item_index, 0, 1)[0]
        except Exception as e:
            logger.debug(f'There is an error: {e}')
            return "I do not know."

        # flatten list because don't want to do preference elicitation
        topk_reviews_flattened_list = reviews[0]
                    
        return self._format_review_resp(
            question, topk_reviews_flattened_list, curr_mentioned_item)

    def _is_category_valid(self, classified_category: str, recommended_item: RecommendedItem) -> bool:
        """
        Returns a bool representing if the classified category is valid.
        A category is valid if it is classified into one of the categories the LLM is told to do.

        :param classified_category: category the LLM classified the user input into
        :param recommended_item: object representing the recommended item the user is referring to
        :return: whether given category is valid
        """
        valid_categories = []

        for key in recommended_item.get_data():
            valid_categories.append(key)        
            
        for valid_category in valid_categories:
            if self._remove_punct_string(valid_category) in self._remove_punct_string(classified_category):
                return True

        return False

    def _format_multiple_qs_resp(self, state_manager: StateManager, all_answers: dict) -> str:
        """
        Returns the response. Returns either an empty string indicating that more work needs to be done to formulate the response or the actual string response.

        :param state_manager: current state representing the conversation
        :param all_answers: list of answers to the users question(s)
        :return: formatted response
        """

        if len(all_answers) > 1:
            user_input = state_manager.get("conv_history")[-1].get_content()

            prompt = self._format_mult_qs_template.render(
                user_input=user_input, all_answers=list(all_answers.values()))
            
            resp = self._llm_wrapper.make_request(prompt)

        else:
            resp = list(all_answers.values())[0]

        return self._clean_llm_response(resp)

    @staticmethod
    def _clean_llm_response(resp: str) -> str:
        """" 
        Clean the response from the llm
        
        :param resp: response from LLM
        :return: cleaned str
        """
        
        if '"' in resp:
            # get rid of double quotes (llm sometimes outputs it)
            resp = resp.replace('"', "")
        
        return resp.removeprefix('Response to user:').removeprefix('response to user:').strip()

    def _format_multiple_item_resp(self, question: str, current_mentioned_items: list[RecommendedItem], answers: dict) \
            -> str:
        """
        Returns the response for one question

        :state_manager: current state representing the conversation
        :param question: the question extracted from the users input
        :param current_mentioned_items: list of current items user is referring to
        :param answers: dict of answers to the users question where the key is the item and the value is the answer to the question
        :return: response for one question
        """
        curr_ment_item_names = [current_mentioned_item.get_name()
                               for current_mentioned_item in current_mentioned_items]

        curr_ment_item_names_str = ", ".join(curr_ment_item_names)

        item_to_answ = {
            ', '.join([f'{key}: {val}' for key, val in answers.items()])}

        if (len(answers) > 1):

            prompt = self._format_mult_resp_template.render(
                question=question, curr_ment_item_names_str=curr_ment_item_names_str,
                res_to_answ=item_to_answ, domain=self._domain)

            resp = self._llm_wrapper.make_request(prompt)

        else:
            resp = list(answers.values())[0]

        if '"' in resp:
            # get rid of double quotes (gpt sometimes outputs it)
            resp = resp.replace('"', "")

        return resp

    def _remove_punct_string(self, expr: str) -> str:
        """ 
        Removes the punctuation, spacing and capitalization from the input. Used to compare strings

        :param expr: string that you want make changes to
        :return: cleaned text
        """
        return ''.join([letter.lower() for letter in expr if letter in ascii_letters])

    def _extract_category_from_input(self, question: str, curr_item: RecommendedItem) -> str:
        """
        Returns the category in the item that can answer the user's question.

        :param question: the question extracted from the users input
        :param curr_item: object representing the recommended item user is referring to
        :return: category corresponding to the user's question
        """

        categories = ""

        for key in curr_item.get_data():
            categories += f" {key},"

        categories += " or none"

        prompt = self._extract_category_template.render(
            curr_item=curr_item, categories=categories, question=question, domain=self._domain,
            few_shots=self._extract_category_few_shots)

        return self._llm_wrapper.make_request(prompt)

    def _create_resp_from_metadata(self, question: str, category: str, recommended_item: RecommendedItem) -> str:
        """
        Returns the string to be returned to the user using metadata

        :param question: the question extracted from the users input
        :param category: category of metadata that information required to answer the question is stored
        :param recommended_item: object representing the reccommended item user is referring to 
        :return: response to user
        """

        for key, val in recommended_item.get_data().items():
            if self._remove_punct_string(key) in self._remove_punct_string(category):

                prompt = self._metadata_template.render(
                    question=question, key=key, val=val)
                
                return self._llm_wrapper.make_request(prompt)

        return ""

    def convert_state_to_query(self, question: str) -> str:
        """
        Returns the string to be returned to be used as the query for information retrieval

        :param question: the question extracted from the users input
        :return: query
        """
        return question

    def _format_review_resp(self, question: str, reviews: list[str], curr_item: RecommendedItem) -> str:
        """
        Returns the string to be returned to the user from the reviews

        :param question: the question extracted from the users input
        :param reviews: the list of reviews corresponding to user input
        :param curr_item: item corresponding the user's question
        :return: response to the user
        """

        try:
            prompt = self._ir_template.render(
                curr_item=curr_item, question=question, reviews=reviews, domain=self._domain,
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

            prompt = self._ir_template.render(
                curr_item=curr_item, question=question, reviews=summarized_reviews, domain=self._domain,
                few_shots=self._ir_prompt_few_shots)

            return self._llm_wrapper.make_request(prompt)

        return resp

    def _notify_observers(self) -> None:
        """
        Notify observers that there are some difficulties.
        """
        for observer in self._observers:
            observer.notify_warning()
