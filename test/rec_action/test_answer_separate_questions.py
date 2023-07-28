import pandas as pd
import pytest
import yaml
import os
from dotenv import load_dotenv
from intelligence.gpt_wrapper import GPTWrapper
from state.common_state_manager import CommonStateManager
from state.message import Message
from rec_action.response_type.answer_prompt_based_resp import AnswerPromptBasedResponse
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper

load_dotenv()


def load_test_data(domain: str) -> list[(str, str)]:
    """
    Load test data using the domain name.

    :param domain: domain name
    :return: test data which is a list od tuple whose first element has a question
             and the second element has individual questions made out of the question
    """
    test_file_path = f'test/rec_action/qa_separate_question_{domain}_test.csv'
    test_df = pd.read_csv(test_file_path)
    test_data = [
        (
            row['question'],
            row['individual_questions'].split("\\n")
        )
        for row in test_df.to_dict("records")]
    return test_data


domain1 = "restaurants"
test_data1 = load_test_data(domain1)

domain2 = "clothing"
test_data2 = load_test_data(domain2)

with open("system_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@pytest.mark.parametrize('llm_wrapper', [GPTWrapper(os.environ['OPENAI_API_KEY']), AlpacaLoraWrapper(os.environ['GRADIO_URL'])])
class TestAnswerSeparateQuestions:

    @pytest.mark.parametrize('question, individual_questions', test_data1)
    def test_separate_question_restaurant(self, llm_wrapper, question: str, individual_questions: str) -> None:
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', question))
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, None,
                                                None, "restaurants", None)
        actual = answer_resp._separate_input_into_multiple_qs(state_manager)
        assert str(actual).lower().strip() \
               == str(individual_questions).lower().strip()

    @pytest.mark.parametrize('question, individual_questions', test_data2)
    def test_separate_question_clothing(self, llm_wrapper, question: str, individual_questions: str) -> None:
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', question))
        answer_resp = AnswerPromptBasedResponse(config, llm_wrapper, None,
                                                None, "clothing", None)
        actual = answer_resp._seperate_input_into_multiple_qs(state_manager)
        assert str(actual).lower().strip() \
               == str(individual_questions).lower().strip()
