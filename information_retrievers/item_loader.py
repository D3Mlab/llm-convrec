from information_retrievers.item import Item
from information_retrievers.recommended_item import RecommendedItem


class ItemLoader:

    def create_item(self, item_dict: dict) -> Item:
        item_id = item_dict.pop('item_id')
        item_name = item_dict.pop('name')
        optional_data = item_dict.pop('optional')
        images = item_dict.pop('imageURLs') if 'imageURLs' in item_dict else []
        return Item(item_id, item_name, item_dict, optional_data, images)

    def create_recommended_item(self, query: str, item_dict: dict, relevant_reviews: list[str]) -> RecommendedItem:
        item = self.create_item(item_dict)
        recommended_item = RecommendedItem(item, query, relevant_reviews)
        return recommended_item
