from information_retriever.item.recommended_item import RecommendedItem
from information_retriever.search_engine.search_engine import SearchEngine
from information_retriever.metadata_wrapper import MetadataWrapper
from information_retriever.item.item_loader import ItemLoader


class InformationRetrieval:
    """
    Information retriever that is responsible for retrieving items and reviews based on query.

    :param search_engine: searches for relevant items and reviews
    :param metadata_wrapper: holds metadata
    :param item_loader: used to load metadata to Item object
    """

    _search_engine: SearchEngine
    _metadata_wrapper: MetadataWrapper
    _item_loader: ItemLoader

    def __init__(self, search_engine: SearchEngine, metadata_wrapper: MetadataWrapper, item_loader: ItemLoader):
        self._search_engine = search_engine
        self._metadata_wrapper = metadata_wrapper
        self._item_loader = item_loader

    def get_best_matching_items(self, query: str, topk_items: int, topk_reviews: int,
                                item_indices_to_keep: list[int], unacceptable_similarity_range: float = 0.5, max_number_similar_items: int = 5) -> list[list[RecommendedItem]]:
        """
        Get k items that match the query the best.

        :param query: query
        :param topk_items: the number of items to return
        :param topk_reviews: the number of reviews to store in a RecommendedItem object
        :param item_indices_to_keep: item indices must be kept
        :param unacceptable_similarity_range: range of similarity scores that would be considered too small to be able to recommend right away
        :param max_number_similar_items: max number of similar items 
        :return: most relevant items and reviews as a list of RecommendedItem objects
        """
        topk_item_id, topk_most_relevant_reviews = \
            self._search_engine.search_for_topk(query, topk_items, topk_reviews, item_indices_to_keep, unacceptable_similarity_range, max_number_similar_items)

        topk_recommended_items_object = self._create_recommended_items(
            query, topk_item_id, topk_most_relevant_reviews)
        return topk_recommended_items_object

    def get_best_matching_reviews_of_item(self, query: str, num_of_reviews_to_return: int,
                                          item_indices_to_keep: list[int], unacceptable_similarity_range: float = 0.5, max_number_similar_items: int = 5) -> list[list[list[str]]]:
        """
        Get num_of_reviews_to_return number of reviews for items that match the query the best.

        :param query: query
        :param num_of_reviews_to_return: the number of reviews to return for each item
        :param item_indices_to_keep: item indices must be kept
        :param unacceptable_similarity_range: range of similarity scores that would be considered too small to be able to recommend right away
        :param max_number_similar_items: max number of similar items 
        :return: most relevant reviews for each item
        """
        topk_items = len(item_indices_to_keep)
        _, topk_most_relevant_reviews = \
            self._search_engine.search_for_topk(query, topk_items, num_of_reviews_to_return, item_indices_to_keep, unacceptable_similarity_range, max_number_similar_items)
        
        return topk_most_relevant_reviews
    
    def _create_recommended_items(self, query: str, item_ids: list[list[str]],
                                  items_most_relevant_reviews: list[list[list[str]]]) -> list[list[RecommendedItem]]:
        """
        Construct RecommendedItem object from query, items id, and most relevant review.

        :param query: query that is used for information retriever
        :param item_ids: items' business id
        :param items_most_relevant_reviews: items' most relevant review stored in the order that corresponds
        to the order of item_ids
        :return: RecommendedItem objects whose business id are in item_bus_id
        """
        recommended_items = []
        for group_index in range(len(item_ids)):
            group_recommended_items = []
            for index in range(len(item_ids[group_index])):
                item_dict = self._metadata_wrapper.get_item_dict_from_id(item_ids[group_index][index])

                recommended_item = self._item_loader.create_recommended_item(query, item_dict,
                                                                            items_most_relevant_reviews[group_index][index])
                group_recommended_items.append(recommended_item)
            
            recommended_items.append(group_recommended_items)
        
        return recommended_items
