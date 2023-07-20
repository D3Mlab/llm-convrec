
class RecommendResponse:
    """
    Class representing the recommend rec action response
    """
    def __init__(self, domain: str):
        self._current_recommended_items = []
        self._domain = domain
        
    def get_current_recommended_items(self):
        return self._current_recommended_items