from information_retriever.item.item import Item
from information_retriever.item.recommended_item import RecommendedItem


class ItemLoader:

    @staticmethod
    def _create_item(item_dict: dict) -> Item:
        """
        Create item object from dict representation of item

        :param item_dict: dict representation of item
        :return: item object corresponding to the given item_dict
        """
        item_id = item_dict.pop('item_id')
        item_name = item_dict.pop('name')
        optional_data = item_dict.pop('optional')
        images = item_dict.pop('imageURLs') if 'imageURLs' in item_dict else []
        return Item(item_id, item_name, item_dict, optional_data, images)

    def create_recommended_item(self, query: str, item_dict: dict, relevant_reviews: list[str]) -> RecommendedItem:
        """
        Create recommended item based on the given query, dict representation of the item and relevant
        reviews of the item.

        :param query: query used to retrieve recommended item
        :param item_dict: dict representation of item
        :param relevant_reviews: reviews of the item relevant to query
        :return: recommended item generated
        """
        item = self._create_item(item_dict)
        recommended_item = RecommendedItem(item, query, relevant_reviews)
        return recommended_item
