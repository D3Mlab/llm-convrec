
class ConstraintMerger:
    """
    Class responsible for merging constraint during constraint update.

    :param constraint: constraint key (e.g. "location") corresponding to this merger
    """

    _constraint: str
    
    def __init__(self, constraint: str):
        self._constraint = constraint
    
    def get_constraint(self) -> str:
        """
        Return the constraint key corresponding to this merger.

        :return: constraint key corresponding to this merger
        """
        return self._constraint

    def merge_constraint(self, og_constraint_value: list[str], new_constraint_value: list[str]) -> list[str]:
        """
        Update the constraint based on the original and new constraint value

        :param og_constraint_value: old constraint value
        :param new_constraint_value: new constraint value
        """
        raise NotImplementedError()
