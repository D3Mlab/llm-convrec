from rec_action.request_information import RequestInformation
from state.common_state_manager import CommonStateManager
from state.state_manager import StateManager
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from rec_action.response_type.request_information_hard_coded_resp import RequestInformationHardCodedBasedResponse
from domain_specific.classes.restaurants.location_status import LocationStatus
import pytest

hard_coded_responses = [{'action': 'RequestInformation',
                         'response': 'Could you provide the cuisine type or dish type?',
                         'constraints': ['cuisine type', 'dish type']},
                        {'action': 'RequestInformation',
                         'response': 'Do you have any other preferences?',
                         'constraints': []}]
request_info_resp = RequestInformationHardCodedBasedResponse([], [LocationStatus(GoogleV3Wrapper())])
request_info = RequestInformation([], hard_coded_responses, request_info_resp)

state_manager1 = CommonStateManager(set())
state_manager1.update("hard_constraints", {"cuisine type": ["dummy_value"], "dish type": ["dummy_value"]})

state_manager2 = CommonStateManager(set())
state_manager2.update("hard_constraints", {"cuisine type": ["dummy_value"]})

state_manager3 = CommonStateManager(set())
state_manager3.update("hard_constraints", {"dish type": ["dummy_value"]})


class TestRequestInformation:

    def test_get_response_with_no_constraints_provided(self) -> None:
        """
        Test whether get_hard_coded_response returns correct response when mandatory constraints
        isn't filled.
        """
        state_manager = CommonStateManager(set())
        state_manager.update("hard_constraints", {})
        expected = "Could you provide the cuisine type or dish type?"
        assert request_info.get_response(state_manager) == expected

    @pytest.mark.parametrize("state_manager", [state_manager1, state_manager2, state_manager3])
    def test_get_response_with_mandatory_constraints_provided(self, state_manager: StateManager) -> None:
        """
        Test whether get_hard_coded_response returns correct response when mandatory constraints
        is filled.

        :param state_manager: current state
        """
        expected = "Do you have any other preferences?"
        assert request_info.get_response(state_manager) == expected
