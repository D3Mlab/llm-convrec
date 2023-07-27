
class RecommendResponse:
    """
    Class representing the recommend rec action response

    domain: domain of the recommendation (e.g. restaurants)
    """
    def __init__(self, domain: str):
        self._current_recommended_items = []
        self._domain = domain
        
    def get_current_recommended_items(self):
        """
        Get most recently recommended items
        """
        return self._current_recommended_items
