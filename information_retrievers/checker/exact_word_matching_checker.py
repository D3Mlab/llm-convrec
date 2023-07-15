from information_retrievers.checker.checker import Checker

class ExactWordMatchingChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether a word in the constraint matches exactly with a word in the specified metadata field
    or a word in the specified metadata field matches exactly with a word in the constraint
    (case insensitive, ignore spaces).
    """

    def check(self, **kwargs):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError