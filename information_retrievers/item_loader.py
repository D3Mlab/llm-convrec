from information_retrievers.item import Item
from information_retrievers.recommended_item import RecommendedItem


class ItemLoader:

    def create_item(self, item_dict: dict) -> Item:
        raise NotImplementedError()

    def create_recommended_item(self, query: str, item_dict: dict, relevant_reviews: list[str]) -> RecommendedItem:
        item = self.create_item(item_dict)
        recommended_item = RecommendedItem(item, query, relevant_reviews)
        return recommended_item
