from information_retrievers.information_retriever import InformationRetriever
from information_retrievers.item import Item
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.neural_ir.neural_search_engine import NeuralSearchEngine
from information_retrievers.filter.filter_restaurants import FilterRestaurants
from information_retrievers.data_holder import DataHolder
import pandas as pd
import torch
import ast


class NeuralInformationRetriever(InformationRetriever):
    """
    Information retriever implemented with constraints filter and 
    BERT_model and item ranking according to the top scoring reviews(late fusion, max) 

    :param engine: searches for relevant items and reviews
    :param data_holder: holds all data needed for information retrieval
    """
    _engine: NeuralSearchEngine
    _data_holder: DataHolder

    def __init__(self, engine: NeuralSearchEngine, data_holder: DataHolder):
        self._engine = engine
        self._data_holder = data_holder

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
            item_object = self._create_item_from_business_id(items_gen_info,
                                                                   item_id[index])
            recommended_item_object = RecommendedItem(item_object, query,
                                                            items_most_relevant_reviews[index])
            recommended_items.append(recommended_item_object)

        return recommended_items

    @staticmethod
    def _create_item_from_business_id(items_gen_info: pd.DataFrame, business_id: str) -> Item:
        """
        Construct Item object from Item's business id

        :param items_gen_info: metadata of Items
        :param business_id: Item's id
        :return: Item objects whose id is business_id
        """
        item_info = items_gen_info.loc[items_gen_info['business_id'] == business_id].iloc[0]
        """
        Item's business id is in the 1st column
        Item's name is in the 2nd column
        Item's address is in the 3rd column
        Item's city is in the 4th column
        Item's state is in the 5th column
        Item's postal code is in the 6th column
        Item's latitude is in the 7th column
        Item's longitude is in the 8th column
        Item's stars is in the 9th column
        Item's review count is in the 10th column
        Item's is open info is in the 11th column
        Item's attributes is in the 12th column
        Item's categories is in the 13th column
        Item's hours is in the 14th column
        """
        business_id = item_info[0]
        dictionary_info = {"name": item_info[1],
                           "address": item_info[2],
                           "city": item_info[3],
                           "state": item_info[4],
                           "postal_code": item_info[5],
                           "latitude": float(item_info[6]),
                           "longitude": float(item_info[7]),
                           "stars": float(item_info[8]),
                           "review_count": int(item_info[9]),
                           "is_open": bool(item_info[10]),
                           "attributes": ast.literal_eval(item_info[11]),
                           "categories": list(item_info[12].split(",")),
                           "hours": ast.literal_eval(item_info[13])}

        item_object = Item(business_id, dictionary_info)
        return item_object
