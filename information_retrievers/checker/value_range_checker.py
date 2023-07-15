from information_retrievers.checker.checker import Checker

class ValueRangeChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether the value in the specified metadata field is within the value range of the constraint.
    """

    def check(self, **kwargs):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError