from information_retrievers.checker.checker import Checker

class WordInChecker(Checker):
    """
    Responsible to check whether the item match the constraint by checking
    whether singular / plural form of a word in the constraint is in the specified metadata field
    or singular / plural form of a word in the specified metadata field is in the constraint.
    """

    def check(self, **kwargs):
        """
        Return true if the item match the constraint, false otherwise.
        """
        raise NotImplementedError