from rec_action.request_information import RequestInformation
from state.common_state_manager import CommonStateManager


class TestRequestInformation:

    def test_get_hard_coded_response_no_mandatory_constraints(self) -> None:
        """
        Test whether get_hard_coded_response returns correct response when mandatory constraints
        isn't filled.
        """
        ri = RequestInformation(None, mandatory_constraints=["dummy1", "dummy2"])
        state_manager = CommonStateManager(set())
        state_manager.update("hard_constraints", {})
        expected = "Can you provide the dummy1?"
        assert ri.get_hard_coded_response(state_manager) == expected

    def test_get_hard_coded_response_with_mandatory_constraints(self) -> None:
        """
        Test whether get_hard_coded_response returns correct response when mandatory constraints
        is filled.
        """
        ri = RequestInformation(None, mandatory_constraints=["dummy1", "dummy2"])
        state_manager = CommonStateManager(set())
        state_manager.update("hard_constraints", {"dummy1": ["dummy_value"], "dummy2": ["dummy_value"]})
        expected = "Are there any additional preferences, requirements, or specific features you would like the " \
                   "restaurant to have?"
        assert ri.get_hard_coded_response(state_manager) == expected
