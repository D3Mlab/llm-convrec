import numpy as np
from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.search_engine.search_engine import SearchEngine
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from information_retrievers.item.item_loader import ItemLoader


class InformationRetrieval:
    """
    Information retriever that is responsible for retrieving items and reviews based on query.

    :param search_engine: searches for relevant items and reviews
    :param metadata_wrapper: holds metadata
    :param item_loader: used to load metadata to Item object
    """

    _engine: SearchEngine
    _metadata_wrapper: MetadataWrapper
    _item_loader: ItemLoader

    def __init__(self, search_engine: SearchEngine, metadata_wrapper: MetadataWrapper,
                 item_loader: ItemLoader) -> None:
        self._search_engine = search_engine
        self._metadata_wrapper = metadata_wrapper
        self._item_loader = item_loader

    def get_best_matching_items(self, query: str, topk_items: int, topk_reviews: int,
                                item_ids_to_keep: np.ndarray) -> list[RecommendedItem]:
        """
        Get k items that match the query the best.

        :param query: query
        :param topk_items: the number of items to return
        :param topk_reviews: the number of reviews to store in a RecommendedItem object
        :param item_ids_to_keep: item ids must be kept
        :return: most relevant items and reviews as a list of RecommendedItem objects
        """
        topk_item_id, topk_most_relevant_reviews = \
            self._search_engine.search_for_topk(query, topk_items, topk_reviews, item_ids_to_keep)

        topk_recommended_items_object = self._create_recommended_items(
            query, topk_item_id, topk_most_relevant_reviews)
        return topk_recommended_items_object

    def get_best_matching_reviews_of_item(self, query: str, num_of_reviews_to_return: int,
                                          item_ids_to_keep: np.ndarray) -> list[list[str]]:
        """
        Get num_of_reviews_to_return number of reviews for items that match the query the best.

        :param query: query
        :param num_of_reviews_to_return: the number of reviews to return for each item
        :param item_ids_to_keep: item ids must be kept
        :return: most relevant reviews for each item
        """
        topk_items = len(item_ids_to_keep)
        _, topk_most_relevant_reviews = \
            self._search_engine.search_for_topk(query, topk_items, num_of_reviews_to_return, item_ids_to_keep)
        return topk_most_relevant_reviews

    def _create_recommended_items(self, query: str, item_ids: list[str],
                                 items_most_relevant_reviews: list[list[str]]) -> list[RecommendedItem]:
        """
        Construct RecommendedItem object from query, items id, and most relevant review.

        :param query: query that is used for information retriever
        :param item_ids: items' business id
        :param items_most_relevant_reviews: items' most relevant review stored in the order that corresponds
        to the order of item_ids
        :return: RecommendedItem objects whose business id are in item_bus_id
        """
        recommended_items = []
        for index in range(len(item_ids)):
            item_dict = self._metadata_wrapper.get_item_dict_from_id(item_ids[index])
            recommended_item = self._item_loader.create_recommended_item(query, item_dict,
                                                                         items_most_relevant_reviews[index])
            recommended_items.append(recommended_item)
        return recommended_items
