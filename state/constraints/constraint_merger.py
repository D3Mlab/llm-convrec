
class ConstraintMerger:
    
    def __init__(self, constraint):
        self._constraint = constraint
    
    def get_constraint(self):
        return self._constraint

    def merge_constraint(self, og_constraint_value, new_constraint_value) -> None:
        """
        Update the constraint based on the original and new constraint value

        :param og_constraint_value: old constraint value
        :param new_constraint_value: new constraint value
        :return: 
        """
        raise NotImplementedError()
