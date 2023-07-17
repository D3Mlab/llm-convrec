from information_retrievers.information_retriever import InformationRetriever
from information_retrievers.item.item import Item
from information_retrievers.item.item_loader import ItemLoader
from information_retrievers.item.recommended_item import RecommendedItem
from information_retrievers.ir.search_engine_old import NeuralSearchEngine
from information_retrievers.data_holder import DataHolder
import pandas as pd
import torch


class NeuralInformationRetriever(InformationRetriever):
    """
    Information retriever implemented with constraints filter and 
    BERT_model and item ranking according to the top scoring reviews(late fusion, max) 

    :param engine: searches for relevant items and reviews
    :param data_holder: holds all data needed for information retrieval
    :param item_loader: used to load metadata to Item object
    """
    _engine: NeuralSearchEngine
    _data_holder: DataHolder
    _item_loader: ItemLoader

    def __init__(self, engine: NeuralSearchEngine, data_holder: DataHolder, item_loader: ItemLoader):
        self._engine = engine
        self._data_holder = data_holder
        self._item_loader = item_loader

    def get_best_matching_items(self, query: str, topk_items: int, topk_reviews: int,
                                filtered_embedding_matrix: torch.Tensor) -> list[RecommendedItem]:
        """
        Get k items that match the query the best.
        
        :param query: query 
        :param topk_items: the number of items to return
        :param topk_reviews: the number of reviews to store in a RecommendedItem object
        :param filtered_embedding_matrix: filtered review embedding matrix
        :return: most relevant items and reviews as a list of RecommendedItem objects
        """
        topk_item_id, topk_items_most_relevant_items = \
            self._engine.search_for_topk(query, topk_items, topk_reviews,
                                         self._data_holder.get_item_reviews_embedding(),
                                         filtered_embedding_matrix,
                                         self._data_holder.get_num_of_reviews_per_item())

        topk_recommended_items_object = self._create_recommended_item(
            query, topk_item_id,
            topk_items_most_relevant_items,
            self._data_holder.get_item_metadata())
        return topk_recommended_items_object

    def get_best_matching_reviews_of_item(self, query: str, item_name: list[str],
                                          num_of_reviews_to_return: int,
                                          items_review_embeddings: pd.DataFrame,
                                          embedding_matrix: torch.Tensor,
                                          num_of_reviews_per_item: torch.Tensor) -> list[list[str]]:
        """
        Get num_of_reviews_to_return number of reviews for items that match the query the best.
        
        :param query: query 
        :param item_name: item's name
        :param num_of_reviews_to_return: the number of reviews to return for each item
        :param items_review_embeddings: items' reviews and review embeddings
        :param embedding_matrix: items' review embedding matrix
        :param num_of_reviews_per_item: number of reviews per item
        :return: most relevant reviews for each item
        """
        topk_items = len(item_name)
        _, topk_item_most_relevant_reviews = \
            self._engine.search_for_topk(query, topk_items, num_of_reviews_to_return,
                                         items_review_embeddings,
                                         embedding_matrix, num_of_reviews_per_item)
        return topk_item_most_relevant_reviews

    def _create_recommended_item(self, query: str, item_id: list[str],
                                 items_most_relevant_reviews: list[list[str]],
                                 items_gen_info: pd.DataFrame) -> list[RecommendedItem]:
        """
        Construct RecommendedItem object from query, items id, and most relevant review.
        
        :param query: query that is used for information retriever
        :param item_id: items' business id
        :param items_most_relevant_reviews: items' most relevant review stored in the order that corresponds
        to the order of item_id
        :param items_gen_info: metadata of items
        :return: RecommendedItem objects whose business id are in item_bus_id
        """
        recommended_items = []
        for index in range(len(item_id)):
            item_dict = items_gen_info.loc[items_gen_info['item_id'] == item_id[index]].to_dict(orient="records")[0]
            recommended_item = self._item_loader.create_recommended_item(query, item_dict,
                                                                         items_most_relevant_reviews[index])
            recommended_items.append(recommended_item)
        return recommended_items

    def _create_item_from_business_id(self, items_gen_info: pd.DataFrame, item_id: str) -> Item:
        """
        Construct Item object from Item's business id

        :param items_gen_info: metadata of Items
        :param item_id: Item's id
        :return: Item objects whose id is item_id
        """
        item_dict = items_gen_info.loc[items_gen_info['item_id'] == item_id].to_dict(orient="records")[0]
        return self._item_loader.create_item(item_dict)
