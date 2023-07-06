
import pytest
import pandas as pd

from intelligence.gpt_wrapper import GPTWrapper
from state.message import Message
from state.constraints.constraints_classifier import ConstraintsClassifier

test_file_path = 'constraints_classifier_test.csv'
test_df = pd.read_csv(test_file_path)
test_data = [
    (
        [row[f'utterance {i}'] for i in range(1, 4) if f'utterance {i}' in row and isinstance(row[f'utterance {i}'], str)],
        {key.removeprefix('input:').strip(): list(map(lambda x: x.lower().strip(), row[key].split(','))) for key in row
         if key.startswith("input:") and isinstance(row[key], str)},
        {'hard_constraints': {key.removeprefix('hard_constraints:').strip(): list(map(lambda x: x.lower().strip(), row[key].split(','))) for key in row
         if key.startswith("hard_constraints:") and isinstance(row[key], str)},
         'soft_constraints': {key.removeprefix('soft_constraints:').strip(): list(map(lambda x: x.lower().strip(), row[key].split(','))) for
         key in row if key.startswith("soft_constraints:") and isinstance(row[key], str)}}
    )
    for row in test_df.to_dict("records")
]
for data in test_data:
    if data[2]['hard_constraints'] == {}:
        data[2].pop('hard_constraints')
    if data[2]['soft_constraints'] == {}:
        data[2].pop('soft_constraints')


class TestConstraintsClassifier:
    @pytest.fixture
    def possible_keys(self):
        yield [key.removeprefix('input:').strip() for key in test_df.head() if key.startswith("input:")]

    @pytest.fixture(params=[GPTWrapper()])
    def classifier(self, request, possible_keys):
        yield ConstraintsClassifier(request.param, possible_keys)

    @pytest.mark.parametrize('utterances,input_constraints,expected_constraints', test_data)
    def test_classify(self, classifier, utterances, input_constraints, expected_constraints) -> None:
        conv_history = []
        role = "user"
        for i in range(len(utterances)):
            conv_history.append(Message(role, utterances[i]))
            role = "recommender" if role == "user" else "user"
        actual_constraints = classifier.classify(conv_history, input_constraints)
        assert actual_constraints == expected_constraints

