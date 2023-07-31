import numpy as np
import torch
from information_retriever.embedder.bert_embedder import BERT_model
from information_retriever.metadata_wrapper import MetadataWrapper
from information_retriever.search_engine.search_engine import SearchEngine


class PDSearchEngine(SearchEngine):
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.

    :param embedder: BERT_model to embed query
    """

    _embedder: BERT_model
    _review_item_ids: np.ndarray
    _reviews: np.ndarray
    _reviews_embedding_matrix: torch.Tensor

    def __init__(self, embedder: BERT_model, review_item_ids: np.ndarray, reviews: np.ndarray,
                 reviews_embedding_matrix: torch.Tensor, metadata_wrapper: MetadataWrapper):
        super().__init__(embedder, review_item_ids, reviews, metadata_wrapper)
        self._reviews_embedding_matrix = reviews_embedding_matrix

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_indices_to_keep: list[int], unacceptable_similarity_range: float, max_number_similar_items: int) -> tuple[list, list]:
        """
        This function takes a query and returns a list of business id that is most similar to the query and the top k
        reviews for that item

        :param query: The input information retriever gets
        :param topk_items: Number of items to be returned
        :param topk_reviews: Number of reviews for each item
        :param item_ids_to_keep: Stores the item id to keep in a numpy array
        :param unacceptable_similarity_range: range of similarity scores that would be considered too small to be able to recommend right away
        :param max_number_similar_items: max number of similar items 
        :return: Return a tuple with element 0 being a list[list[str]] which is a list of similar items item_id (similar items are items where their similarity score is less than unacceptable similarity range)
        element 1 being list[list[list[str]]] with Dim 0 has the group of similar items, Dim 1 has the similar item, Dim 2 has the top k
        reviews for the corresponding item
        """
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._similarity_score_each_review(
            query_embedding)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(
            similarity_score_review, topk_reviews)
        similarity_score_item = self._filter_item_similarity_score(similarity_score_item, item_indices_to_keep)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items, unacceptable_similarity_range, max_number_similar_items)
        list_of_item_id = self._get_topk_item_id(most_similar_item_index, index_most_similar_review)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review)

        return list_of_item_id, list_of_review

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        This function finds and returns a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """
        similarity_score = torch.matmul(self._reviews_embedding_matrix, query)
        return similarity_score
