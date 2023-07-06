import pytest
import pandas as pd
from intelligence.gpt_wrapper import GPTWrapper
from state.message import Message
from state.constraints.key_value_pair_constraints_extractor import KeyValuePairConstraintsExtractor

test_file_path = 'constraints_extractor_test_v2.csv'
test_df = pd.read_csv(test_file_path)
test_data = [
    (
        [row[f'utterance {i}'] for i in range(1, 4) if f'utterance {i}' in row and isinstance(row[f'utterance {i}'], str)],
        {key: list(map(lambda x: x.lower().strip(), row[key].split(','))) for key in row if
         not key.startswith("utterance")
         and isinstance(row[key], str)}
    )
    for row in test_df.to_dict("records")
]


class TestConstraintsExtractor:

    @pytest.fixture
    def possible_keys(self):
        constraints = ["location", "cuisine type", "dish type", "type of meal", "price range", "wait times", "atmosphere",
                       "dietary restrictions", "others"]
        yield constraints

    @pytest.fixture(params=[GPTWrapper()])
    def extractor(self, request, possible_keys):
        constraint_descriptions = [
            'The desired location of the restaurants.',
            'The desired specific style of cooking or cuisine offered by the restaurants (e.g., "Italian", "Mexican", "Chinese"). This can be implicitly provided through dish type (e.g "italian" if dish type is "pizza")',
            'The desired menu item or dish in the restaurant that user shows interests.',
            'The desired category of food consumption associated with specific times of day (e.g., "breakfast", "lunch", "dinner").',
            'The preferred range of prices for the restaurants as specified by the user.',
            'The acceptable wait time for the user when dining at the restaurants.',
            'The preferred atmosphere or ambience of the restaurants.',
            'Any specific dietary limitations or restrictions the user may have.',
            'Any additional constraints or preferred features (e.g. "patio", "free wifi", "free parking", ...).',
        ]
        yield KeyValuePairConstraintsExtractor(request.param, possible_keys, constraint_descriptions)

    @pytest.mark.parametrize('utterances,expected', test_data)
    def test_extract(self, extractor, utterances, expected) -> None:
        conv_history = []
        role = "user"
        for i in range(len(utterances)):
            conv_history.append(Message(role, utterances[i]))
            role = "recommender" if role == "user" else "user"
        actual = extractor.extract(conv_history)
        assert actual == expected
