import logging
from rec_action.rec_action import RecAction
from state.state_manager import StateManager
from user_intent.inquire import Inquire
from state.message import Message
from decimal import Decimal
from string import ascii_letters
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.neural_information_retriever import InformationRetriever
from intelligence.llm_wrapper import LLMWrapper
from domain_specific_config_loader import DomainSpecificConfigLoader
from jinja2 import Environment, FileSystemLoader
import yaml

logger = logging.getLogger('answer')


class Answer(RecAction):
    """
    Class representing Answer recommender action.
    :param config: config values from system_config.yaml
    :param priority_score_range: range of scores for smth TODO: fill in smth
    :param information_retriever: information retriever that is used to fetch restaurant recommendations
    :param llm_wrapper: object to make request to LLM
    """

    _num_of_reviews_to_return: int
    _filter_restaurants: FilterRestaurants
    _information_retriever: InformationRetriever
    _llm_wrapper: LLMWrapper
    _prompt: str

    def __init__(self, config: dict, llm_wrapper: LLMWrapper, filter_restaurants: FilterRestaurants,
                 information_retriever: InformationRetriever, domain: str,
                 priority_score_range: tuple[float, float] = (1, 10)) -> None:
        super().__init__(priority_score_range)
        self._filter_restaurants = filter_restaurants
        self._domain = domain

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

        self.env = Environment(loader=FileSystemLoader(
            self.config['ANSWER_PROMPTS_PATH']), trim_blocks=True, lstrip_blocks=True)

        self.gpt_template = self.env.get_template(
            self.config['ANSWER_GPT_PROMPT'])

        self.mult_qs_template = self.env.get_template(
            self.config['ANSWER_MULT_QS_PROMPT'])

        self.verify_metadata_template = self.env.get_template(
            self.config['ANSWER_VERIFY_METADATA_RESP_PROMPT'])

        self.format_mult_qs_template = self.env.get_template(
            self.config['ANSWER_MULT_QS_FORMAT_RESP_PROMPT'])

        self.format_mult_resp_template = self.env.get_template(
            self.config['ANSWER_FORMAT_MULTIPLE_RESP_PROMPT'])

        self.extract_category_template = self.env.get_template(
            self.config['ANSWER_EXTRACT_CATEGORY_PROMPT'])

        self.hours_template = self.env.get_template(
            self.config['ANSWER_HOURS_PROMPT'])

        self.attr_template = self.env.get_template(
            self.config['ANSWER_ATTR_PROMPT'])

        self.ir_template = self.env.get_template(
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

    def get_prompt(self, state_manager: StateManager) -> str | None:
        """
        Return prompt that can be inputted to LLM to produce recommender's response. 
        Return None if it doesn't exist. 

        :param state_manager: current state representing the conversation
        :return: prompt that can be inputted to LLM to produce recommender's response or None if it doesn't exist. 
        """
        return self._prompt

    def get_hard_coded_response(self, state_manager: StateManager) -> str | None:
        """
        Return hard coded recommender's response corresponding to this action. 

        :param state_manager: current state representing the conversation
        :return: hard coded recommender's response corresponding to this action
        """

        curr_mentioned_restaurants: list[RecommendedItem] = state_manager.get(
            "curr_items")

        answer_all_q = []

        if curr_mentioned_restaurants is not None:

            user_questions = self._seperate_input_into_multiple_qs(
                state_manager)

            for question in user_questions:
                logger.debug(
                    f'The question is {question}')

                answer_one_q = {}
                llm_resp = ""

                for curr_mentioned_restaurant in curr_mentioned_restaurants:
                    logger.debug(
                        f'The recommended restaurant is {curr_mentioned_restaurant.get("name")}')
                    category = self._extract_category_from_input(
                        question, curr_mentioned_restaurant)

                    logger.debug(f'The category is {category}')

                    logger.debug(
                        f'Is the category the LLM classified the user input into valid: {self._is_category_valid(category, curr_mentioned_restaurant)}')

                    if self._is_category_valid(category, curr_mentioned_restaurant):
                        logger.debug("Metadata question!")

                        metadata_resp = (self._create_resp_from_metadata(
                            question, category, curr_mentioned_restaurant))

                        answer_one_q[curr_mentioned_restaurant.get("name")
                                     ] = metadata_resp

                    if not self._is_category_valid(category, curr_mentioned_restaurant) or metadata_resp == "" or not self._verify_metadata_resp(question, metadata_resp):

                        logger.debug("Non metadata question!")

                        ir_resp = self._create_resp_from_ir(
                            question, curr_mentioned_restaurant)

                        answer_one_q[curr_mentioned_restaurant.get("name")
                                     ] = ir_resp

                        if "I do not know" in ir_resp:
                            logger.debug(
                                f'Answer with GPT')

                            prompt = self.gpt_template.render(
                                curr_mentioned_item=curr_mentioned_restaurant, question=question)

                            llm_resp = self._llm_wrapper.make_request(
                                prompt)

                            answer_one_q[curr_mentioned_restaurant.get("name")
                                         ] = llm_resp

                mult_rest_resp = self._format_multiple_restaurant_resp(
                    question, curr_mentioned_restaurants, answer_one_q)

                # If only one question then add note to beginning of response
                if llm_resp != "" and len(user_questions) == 1:
                    mult_rest_resp = "I couldn't find any relevant information in the product database to help me respond. Based on my internal knowledge, which does not include any information after 2021..." + '\n' + mult_rest_resp

                answer_all_q.append(mult_rest_resp)

        else:
            return "Please ask questions about previously recommended restaurants."

        return self._format_multiple_qs_resp(state_manager, answer_all_q, llm_resp != "")

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

    def _create_resp_from_ir(self, question: str, curr_mentioned_restaurant: RecommendedItem):
        """
        Returns the string to be returned to the user when using information retrieval
        :param question: the question extracted from the users input
        :param curr_mentioned_restaurant: one of the recommended restaurants user is currently referring to
        :returns: resp to user.
        """
        query = self.convert_state_to_query(
            question, curr_mentioned_restaurant)

        logger.debug(f'Query: {query}')

        filtered_embedding_matrix, filtered_num_of_reviews_per_restaurant, \
            filtered_restaurants_review_embeddings = \
            self._filter_restaurants.filter_by_restaurant_name(
                [curr_mentioned_restaurant.get("name")])
        reviews = self._information_retriever.get_best_matching_reviews_of_item(
            query, [curr_mentioned_restaurant.get("name"
                                                  )], self._num_of_reviews_to_return,
            filtered_restaurants_review_embeddings,
            filtered_embedding_matrix, filtered_num_of_reviews_per_restaurant)[0]

        return self._format_review_resp(
            question, reviews, curr_mentioned_restaurant)

    def _is_category_valid(self, classified_category: str, recommended_restaurant: RecommendedItem) -> bool:
        """
        Returns a bool representing if the classified category is valid.
        A cateogry is valid if it is classified into one of the categories the LLM is told to do.
        :param classified_category: cateogry the LLM classified the user input into
        :param recommended_restaurant: object representing the reccommended restaurant user is referring to
        :return: bool
        """
        valid_categories = ["address", "city", "state",
                            "postal_code", "stars", "review_count", "hours"]

        for key in recommended_restaurant.get("attributes"):
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

    def _format_multiple_qs_resp(self, state_manager: StateManager, all_answers: list[str], is_llm_res: bool) -> str:
        """
        Returns the response. Returns either an empty string indicating that more work needs to be done to formulate the response or the actual string response.

        :state_manager: current state representing the conversation
        :param all_answers: list of answers to the users question(s)
        :param is_llm_res: boolean indicating if the answer was created using an LLM or not
        :return: str
        """

        user_input = state_manager.get("conv_history")[-1].get_content()
        resp = ""

        if (len(all_answers) > 1):

            self._prompt = self.format_mult_qs_template.render(
                user_input=user_input, all_answers=all_answers)

            if is_llm_res:
                resp = "I couldn't find any relevant information in the product database to help me respond. Based on my internal knowledge, which does not include any information after 2021..." + '\n'

        else:
            resp = all_answers[0]

            if '"' in resp:
                # get rid of double quotes (gpt sometimes outputs it)
                resp = resp.replace('"', "")

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

        curr_ment_res_names = [current_mentioned_restaurant.get('name'
                                                                ) for current_mentioned_restaurant in current_mentioned_restaurants]

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

        categories = "address, city, state, postal_code, stars, review_count, hours,"

        for key in curr_restaurant.get("attributes"):
            if key == 'GoodForMeal':
                categories += f" best meals,"
            else:
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

        if "address" in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')} is located at {recommended_restaurant.get('address')} in {recommended_restaurant.get('city')}, {recommended_restaurant.get('state')}."

        elif "city" in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')} is in {recommended_restaurant.get('city')}."

        elif 'state' in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')} is in {recommended_restaurant.get('state')}."

        elif self._remove_punct_string('postal_code') in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')}'s postal code is {recommended_restaurant.get('postal_code')}."

        elif "stars" in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')} has a rating of {recommended_restaurant.get('stars')} / 5."

        elif self._remove_punct_string('review_count') in self._remove_punct_string(category):
            resp = f"{recommended_restaurant.get('name')} has {recommended_restaurant.get('review_count')} reviews."

        elif 'hours' in self._remove_punct_string(category):

            hours = recommended_restaurant.get('hours')

            resp = f"{recommended_restaurant.get('name')} is "

            for key, val in hours.items():
                if val == '0:0-0:0':
                    resp += f'not open on {key}, '
                else:
                    val = val.replace(':', '.')

                    try:
                        time_open, time_close = val.split('-')
                    except:

                        prompt = self.hours_template.render(
                            question=question, recommended_item=recommended_restaurant)

                        return self._llm_wrapper.make_request(prompt)

                    time_open, time_close = self._format_time_open_and_close(
                        time_open, time_close)

                    resp += f'is open on {key} from {time_open} to {time_close}, '

            resp = f'{resp[:-2]}.'

        else:
            for key, val in recommended_restaurant.get("attributes").items():
                if self._remove_punct_string(key) in self._remove_punct_string(category):

                    prompt = self.attr_template.render(
                        question=question, key=key, val=val)

                    resp = self._get_resp_from_attr(
                        key, val, recommended_restaurant.get('name'), prompt)

        return resp

    def _get_resp_from_attr(self, key: str, val: str | dict, restaurant_name: str, prompt: str) -> str:
        """ 
        Get response if metadata category is an attribute

        :param key: the attribute category (ex. HasRestaurant)
        :param value: the value of the attribute category (ex. True)
        :param restaurant_name: the name of the restaurant
        :param prompt: prompt to use if cannot get the metadata response
        :returns: metadata response
        """
        dict_val = {}
        str_val = val.split()[0]

        # if value is a dictionary, assume there is no nested dictionarys and just has key and values which are all strings.
        if '{' in val:
            val = val.replace('{', '')
            val = val.replace('}', '')

            try:
                val_list = val.split(',')
            except:
                return self._llm_wrapper.make_request(prompt)

            for single_attr in val_list:
                try:
                    key_single_attr, val_single_attr = single_attr.split(':')
                except:
                    return self._llm_wrapper.make_request(prompt)

                dict_val[key_single_attr.strip()] = val_single_attr.strip()

        # format response if value is a string
        # TODO: see if its in the acceptable range of values and if not them prompt to gpt  -> range of values = {'very_loud', 'average', '1', 'no', '3', 'full_bar', 'casual', '4', 'False', 'quie't, 'None', 'none', '2', 'True', 'paid', 'free', 'formal', 'loud', 'outdoor', 'dressy', 'beer_and_wine'}

        if dict_val == {}:
            # if string is not valid then return empty string
            if not self._valid_keys_for_str_attr(key, str_val):
                return self._llm_wrapper.make_request(prompt)

            resp = self._get_attr_resp_from_str(
                key, str_val, restaurant_name, prompt)

        # format response if value is the dictionary
        if dict_val != {}:
            # If keys and values are not valid then return empty string
            if not self._valid_keys_for_dict_attr(key, dict_val):
                return self._llm_wrapper.make_request(prompt)

            resp = self._get_attr_resp_from_dict(
                key, dict_val, restaurant_name)

        return resp

    def _extract_resp_str(self, dict_val: dict, attribute: str, val_to_return_str: bool, str_to_return: str) -> str:
        """ 
        If metadata attribute category has a dict type value then extract the individual values in the dict or return an empty string if it doesn't exist

        :param dict_val: dictionary of the metadata value  ex. {dj: False, live: False, jukebox: None, video: False}
        :param attribute: the key in dict_val you want the value of (ex. dj)
        :returns: the value from dict_val (ex. False)
        """
        if dict_val.get(attribute) is not None and dict_val.get(attribute) != 'None' and dict_val.get(attribute) == val_to_return_str:
            return str_to_return
        else:
            return ""

    def _has_expected_keys_and_values(self, expected_keys: list, attr_dict: dict, range_expected_values=["False", "True", "None"]) -> bool:
        """ 
        See if the keys and values in the metadata attribute category's dictionary are valid. Helper function for _valid_keys_for_dict_attr(self, curr_key, dict_val)

        :param attr_dict: dictionary of the metadata value  ex. {dj: False, live: False, jukebox: None, video: False}
        :param expected_keys: the keys that you would expect in attr_dict (ex. ["dj", "live"])
        :param range_expected_values: list representing the expected values in attr_dict
        :returns: boolean if it has the expected keys and values
        """
        for key, val in attr_dict.items():
            if key not in expected_keys:
                logger.debug(f'Foreign key is {key} in dict {attr_dict}')
                return False

            if val not in range_expected_values:
                logger.debug(f'Foreign val is {val} in dict {attr_dict}')
                return False

    def _valid_keys_for_dict_attr(self, curr_key, dict_val) -> bool:
        """ 
        If metadata attribute category's value is a dictionary then see if the keys and values in the metadata attribute category's dictionary are valid

        :param curr_key: metadata attribute category
        :param dict_val: dictionary of the metadata value  ex. {dj: False, live: False, jukebox: None, video: False}
        :returns: boolean if it has the expected keys and values
        """
        if curr_key == 'Music':
            expected_music_keys = [
                'dj', 'live', 'jukebox', 'video', 'background_music', 'karaoke', 'no_music']
            if (self._has_expected_keys_and_values(expected_music_keys, dict_val)):
                return True
            else:
                return False

        elif curr_key == 'Ambience':
            expected_ambience_keys = [
                'touristy', 'hipster', 'romantic', 'intimate', 'trendy', 'upscale', 'classy', 'casual', 'divey']
            if (self._has_expected_keys_and_values(expected_ambience_keys, dict_val)):
                return True
            else:
                return False

        elif curr_key == 'PopularNights':
            expected_best_nights_keys = [
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            if (self._has_expected_keys_and_values(expected_best_nights_keys, dict_val)):
                return True
            else:
                return False

        elif curr_key == 'GoodForMeal':
            expected_good_meal_keys = [
                'dessert', 'lunch', 'dinner', 'brunch', 'breakfast', 'latenight']
            if (self._has_expected_keys_and_values(expected_good_meal_keys, dict_val)):
                return True
            else:
                return False

        elif curr_key == 'BusinessParking':
            expected_parking_keys = [
                'garage', 'street', 'validated', 'lot', 'valet']
            if (self._has_expected_keys_and_values(expected_parking_keys, dict_val)):
                return True
            else:
                return True

        elif curr_key == 'DietaryRestrictions':
            expected_dietary_restrictions_keys = ['dairy-free', 'gluten-free', 'vegan',
                                                  'kosher', 'halal', 'soy-free', 'vegetarian']
            if (self._has_expected_keys_and_values(expected_dietary_restrictions_keys, dict_val)):
                return True
            else:
                return False

        # invalid key
        else:
            return False

    def _valid_keys_for_str_attr(self, curr_key, val) -> bool:
        """ 
        Determines if metadata attribute key is valid or not for values that are strings

        :param key: the attribute category (ex. HasRestaurant)
        :param value: the value of the attribute category (ex. True)
        :returns: bool representing if metadata attribute key is valid
        """
        valid_keys = ['PriceRange', 'Alcohol', 'WiFi', 'GoodForGroups', 'MustMakeReservation', 'BusinessAcceptsCreditCards', 'HasDelivery', 'HasReservations', 'Smoking', 'HasTakeOut',
                      'WheelchairAccessible', 'GoodForKids', 'HappyHour', 'DogsAllowed', 'GoodForDancing', 'BYOB', 'HasTV', 'NoiseLevel', 'BikeParking', 'HasTableService', 'Corkage', 'CoatCheck', 'OutdoorSeating', 'DriveThru', 'Attire', 'Caters']

        if curr_key not in valid_keys and val != 'None':
            logger.debug(f'Foreign key is {curr_key}')
            return False

        return True

    def _format_time_open_and_close(self, time_open: str, time_close: str) -> tuple:
        """ 
        If metadata key is hours then format the time the restaurant opens and time the restaurant closes.

        :param time_open: time the restaurant opens
        :param time_close: time the restaurant closes
        :returns: new time open and time close values
        """
        if time_open[::-1].find('.') == 1:
            time_open += '0'

        if time_close[::-1].find('.') == 1:
            time_close += '0'

        if float(time_open) > 12.0:
            time_open = str(Decimal(time_open) -
                            Decimal(12.0)) + 'pm'
        elif time_open == '0.00':
            time_open = '12.00am'
        else:
            time_open += 'am'

        if float(time_close) > 12.0:
            time_close = str(Decimal(time_close) -
                             Decimal(12.00)) + 'pm'
        elif time_close == '0.00':
            time_close = '12.00am'
        else:
            time_close += 'am'

        time_open = time_open.replace('.', ':')
        time_close = time_close.replace('.', ':')

        return time_open, time_close

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
            print(
                'Sorry.. running into some difficulties, this is going to take longer than ususal.')

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

    def _get_attr_resp_from_str(self, key: str, str_val: str, restaurant_name: str, prompt: str) -> str:
        """
        Gets the metadata response for an attribute type key and the value is a string

        :param key: the metadata attribute category (ex. GoodForMeal)
        :param str_val: The value of the metadata attribute category (ex. False)
        :param restaurant_name: the name of the restaurant
        :param prompt: the prompt to use if you have an unrecognized value 

        :return: response
        """
        resp = f"{restaurant_name} "

        # underscore to two words
        if '_' in str_val:
            str_val = " ".join(str_val.split('_'))

        if key == 'NoiseLevel' and str_val != 'None' and str_val != 'none':
            resp += 'has a'
            if str_val == 'average':
                resp += "n "
            else:
                resp += " "

            resp += f'{str_val} noise level.'

        elif key == 'PriceRange':
            if str_val == 'None':
                resp = f"{restaurant_name}'s price range is unknown."
            else:
                resp = f"{restaurant_name} has a price range of {str_val} / 4, where 1 is the cheapest option (less than $10 per person) and 4 is the most expensive (greater than $61 per person)."

            if str_val == '2':
                resp += ' A rating of 2 means that the average person spends $11-$30 per person.'

            if str_val == '3':
                resp += ' A rating of 3 means that the average person spends $31-$60 per person.'

        elif key == 'Alcohol':
            if str_val == 'None' or str_val == 'none':
                resp += 'does not serve any alcohol.'

            elif str_val == 'beer and wine':
                resp += f'serves {str_val} but does not have a full bar'
            else:
                resp += f'has a {str_val}.'

        elif key == 'WiFi':
            if str_val == 'None':
                resp = f"It is unknown if {restaurant_name} has WiFi."
            else:
                resp += f"has {str_val} WiFi."

        elif key == 'GoodForGroups':
            if str_val == 'True':
                resp += f"is good for groups."
            elif str_val == 'False':
                resp += f"is not good for groups."
            else:
                resp = f"It is unknown if {restaurant_name} is good for groups."

        elif key == 'MustMakeReservation':
            if str_val == 'True':
                resp = f"You must make a reservation to go to {restaurant_name}."
            else:
                resp = f"You do not need a reservation to go to {restaurant_name}."

        elif key == 'BusinessAcceptsCreditCards':
            if str_val == 'True':
                resp += "does accept credit cards."
            elif str_val == 'False':
                resp += "does not accept credit cards."
            else:
                resp = f"I don't know if {restaurant_name} accepts credit cards."

        elif key == 'HasDelivery':
            if str_val == 'True':
                resp += "does have delivery."
            elif str_val == 'False':
                resp += "does not have delivery."
            else:
                resp = f"I am not sure if {restaurant_name} has delivery."

        elif key == "HasReservations":
            if str_val == 'True':
                resp += "accepts reservations."
            elif str_val == 'False':
                resp += "does not accept reservations."
            else:
                resp = f"I am not sure if {restaurant_name} accepts reservations."

        elif key == "Smoking":
            if str_val == 'outdoor':
                resp += "has outdoor smoking."
            elif str_val == "no":
                resp += "has a no smoking policy."
            else:
                # If it is a value don't recognize return llm response of prompt
                return

        elif key == 'HasTakeOut':
            if str_val == 'False':
                resp += "does not have takeout."
            elif str_val == "True":
                resp += "does have takeout."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has takeout."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "WheelchairAccessible":
            if str_val == 'False':
                resp += "is not wheelchair accessible."
            elif str_val == "True":
                resp += "is wheelchair accessible."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "GoodForKids":
            if str_val == 'False':
                resp += "is not kid friendly."
            elif str_val == "True":
                resp += "is kid friendly."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} is good for kids."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "HappyHour":
            if str_val == 'False':
                resp += "does not have a happy hour."
            elif str_val == "True":
                resp += "does have a happy hour."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has a happy hour."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "DogsAllowed":
            if str_val == 'False':
                resp += "does not allow dogs inside."
            elif str_val == "True":
                resp += "does allow dogs inside."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} allows dogs."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "GoodForDancing":
            if str_val == 'False':
                resp += "is not a good place to go dancing."
            elif str_val == "True":
                resp += "is a good place to go dancing."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "BYOB":
            if str_val == 'False':
                resp += "is not BYOB."
            elif str_val == "True":
                resp += "is BYOB."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} is BYOB."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "HasTV":
            if str_val == 'False':
                resp += "does not have a TV."
            elif str_val == "True":
                resp += "has a TV."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "BikeParking":
            if str_val == 'False':
                resp += "does not have bike parking."
            elif str_val == "True":
                resp += "does have bike parking."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has bike parking."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "HasTableService":
            if str_val == 'False':
                resp += "does not have table service."
            elif str_val == "True":
                resp += "does have table service."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has table service."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "Corkage":
            if str_val == 'False':
                resp += "does not have a corkage fee."
            elif str_val == "True":
                resp += "does have a corkage fee."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "CoatCheck":
            if str_val == 'False':
                resp += "does not have coat check."
            elif str_val == "True":
                resp += "does have coat check."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "OutdoorSeating":
            if str_val == 'False':
                resp += "does not have outdoor seating."
            elif str_val == "True":
                resp += "does have outdoor seating."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has outdoor seating."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "DriveThru":
            if str_val == 'False':
                resp += "does not have a drive thru."
            elif str_val == "True":
                resp += "does have a drive thru."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        elif key == "Attire":
            if str_val == "None":
                resp = f"I am not sure what {restaurant_name}'s dress code is."
            else:
                resp += f"has a {str_val} dress code."

        elif key == "Caters":
            if str_val == 'False':
                resp += "does not have catering."
            elif str_val == "True":
                resp += "does have catering."
            elif str_val == "None":
                resp = f"I am not sure if {restaurant_name} has catering."
            else:
                # If it is a value don't recognize return llm response of prompt
                return self._llm_wrapper.make_request(prompt)

        return resp

    def _get_attr_resp_from_dict(self, key: str, dict_val: str, restaurant_name: str) -> str:
        """
        Gets the metadata response for an attribute type key and the value is a dict

        :param key: the metadata attribute category (ex. Ambience)
        :param dict_val: The value of the metadata attribute category (ex. {'dj': 'False', 'live': 'True'})
        :param restaurant_name: the name of the restaurant

        :return: response
        """
        resp = f"{restaurant_name} "

        if key == 'Music':
            resp += self._extract_resp_str(dict_val, 'no_music',
                                           'True', 'does not have music, ')
            resp += self._extract_resp_str(dict_val, 'no_music', 'False',
                                           f"does have music. In fact, {restaurant_name} does have ")

            music_attr_to_resp_str = {
                'dj': 'a DJ, ',
                'live': 'live music, ',
                'jukebox': 'a jukebox, ',
                'video': 'a TV playing music videos, ',
                'background_music': 'background music, ',
                'karaoke': 'karaoke, '
            }

            for music_attr, resp_str in music_attr_to_resp_str.items():
                resp += self._extract_resp_str(dict_val,
                                               music_attr, 'True', resp_str)

            transition_word = ""
            if resp.endswith(f"{restaurant_name} does have "):
                resp = resp[:-36]
                transition_word = " But "
            elif resp.endswith('does not have music, '):
                resp = f'{resp[:-2]}. '
                transition_word = "More specifically, "
            else:
                resp = f'{resp[:-2]}. '
                transition_word = "However, "

            resp += transition_word + "doesn't have "

            for music_attr, resp_str in music_attr_to_resp_str.items():
                resp += self._extract_resp_str(dict_val,
                                               music_attr, 'False', resp_str)

            if resp.endswith("doesn't have "):
                index_rmv = len("doesn't have ") + len(transition_word)
            else:
                index_rmv = 2

            resp = resp[:-index_rmv] + '.'

        elif key == 'Ambience':
            ambience_keys = [
                'touristy', 'hipster', 'romantic', 'intimate', 'trendy', 'upscale', 'classy', 'casual', 'divey']
            resp += 'has a '

            for ambience_attr in ambience_keys:
                resp += self._extract_resp_str(dict_val, ambience_attr,
                                               'True', f'{ambience_attr}, ')
            if resp == f"{restaurant_name} has a ":
                resp = "I am not sure what the atmosphere of resaurant_name is but, I know it doesn't have a "

                for ambience_attr in ambience_keys:
                    resp += self._extract_resp_str(dict_val, ambience_attr,
                                                   'False', f'{ambience_attr}, ')

            resp = resp[:-2]
            resp += ' vibe.'

        elif key == 'PopularNights':
            best_nights_keys = [
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

            resp += "has their best nights on "

            for best_night_attr in best_nights_keys:
                resp += self._extract_resp_str(dict_val, best_night_attr,
                                               'True', f"{best_night_attr}'s, ")

            resp = resp[:-2] + '.'
            if resp == f"{restaurant_name} 's best nights ar.":
                resp = f"I do not have any information on {restaurant_name}'s best nights."

        elif key == 'GoodForMeal':
            good_meal_keys = [
                'dessert', 'lunch', 'dinner', 'brunch', 'breakfast']

            resp += 'is a popular spot for '

            simple_resp_str = good_meal_keys[:-1]

            for best_night_attr in simple_resp_str:
                resp += self._extract_resp_str(dict_val, best_night_attr,
                                               'True', f"{best_night_attr}, ")

            resp += self._extract_resp_str(dict_val, 'latenight',
                                           'True', f"late night meals, ")

            resp = resp[:-2] + '.'

            if resp == f"{restaurant_name} is a popular spot fo.":
                resp = f"From my knowledge, {restaurant_name} does not have particular meals in the day where it is very popular."

        elif key == 'BusinessParking':
            parking_keys_to_resp_str = {
                'garage': 'a parking garage, ',
                'lot': 'a parking lot, ',
                'street': 'street parking, ',
                'valet': 'valet parking, ',
            }

            resp += 'has '

            for parking_key, resp_str in parking_keys_to_resp_str.items():
                resp += self._extract_resp_str(dict_val, parking_key,
                                               'True', resp_str)

            resp += self._extract_resp_str(dict_val, 'validated',
                                           'True', "and has validated parking, ")

            resp = resp[:-2] + '.'

            if resp == f"{restaurant_name} ha.":
                resp = f"{restaurant_name} does not have parking. "

        else:
            dietary_restrictions_keys = ['dairy-free', 'gluten-free', 'vegan',
                                         'kosher', 'halal', 'soy-free', 'vegetarian']
            resp += "has "

            for dietary_restriction_key in dietary_restrictions_keys:
                resp += self._extract_resp_str(dict_val, dietary_restriction_key,
                                               'True', f'{dietary_restriction_key}, ')

            if resp == f"{restaurant_name} has ":
                resp = f"According to my database, {restaurant_name} does not have "

                for dietary_restriction_key in dietary_restrictions_keys:
                    resp += self._extract_resp_str(dict_val, dietary_restriction_key,
                                                   'False', f'{dietary_restriction_key}, ')

            resp = resp[:-2] + ' options.'

        return resp
