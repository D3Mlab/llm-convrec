from information_retriever.item.item import Item


class RecommendedItem(Item):
    """
    Class represents a recommended item

    :param item: Item object that will be used to construct RecommendedItem object
    :param query: query that is used to retrieve this item
    :param most_relevant_reviews: reviews of the restaurant that are most relevant to the query
    """

    _query: str
    _most_relevant_reviews: list[str]

    def __init__(self, item: Item, query: str, most_relevant_reviews: list[str]):
        super().__init__(item.get_id(), item.get_name(), item.get_mandatory_data(), item.get_optional_data())
        self._query = query
        self._most_relevant_reviews = most_relevant_reviews

    def get_query(self) -> str:
        """
        Get the query.
        
        :return: query stored in the object
        """
        return self._query

    def get_most_relevant_review(self) -> list[str]:
        """
        Get the most relevant reviews.
        
        :return: most relevant reviews stored in the object
        """
        return self._most_relevant_reviews
