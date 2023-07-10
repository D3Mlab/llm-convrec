import pandas as pd
import pytest
import yaml

from intelligence.gpt_wrapper import GPTWrapper
from rec_action.answer import Answer
from state.common_state_manager import CommonStateManager
from state.message import Message

domain = "restaurants"
test_file_path = 'test/rec_action/qa_separate_question_test.csv'
test_df = pd.read_csv(test_file_path)
test_data = [
    (
        row['question'],
        row['individual_questions'].split("\\n")
    )
    for row in test_df.to_dict("records")]

class TestAnswer:

    @pytest.fixture(params=[GPTWrapper()])
    def answer(self, request):
        with open("system_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        yield Answer(config, request.param, None, None, domain)

    @pytest.mark.parametrize('question,individual_questions', test_data)
    def test_separate_question(self, answer, question, individual_questions) -> None:
        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', question))
        actual = answer._seperate_input_into_multiple_qs(state_manager)
        assert str(actual).lower().replace(" ", "").replace("\\", "") \
               == str(individual_questions).lower().replace(" ", "").replace("\\", "")