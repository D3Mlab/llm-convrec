from information_retrievers.item import Item


class ItemLoader:

    def create_item(self, item_dict: dict) -> Item:
        raise NotImplementedError()

    def create_recommended_item(self, query: str, item_dict: dict, relevant_reviews: list[str]) -> Item:
        pass

