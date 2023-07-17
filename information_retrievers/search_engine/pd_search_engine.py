import numpy as np
import torch
import pandas as pd
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.search_engine.search_engine import SearchEngine


class PDSearchEngine(SearchEngine):
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.

    :param embedder: BERT_model to embed query
    :param path_to_items_id: Stores the path towards item id numpy array
    :param path_to_items_review_embeddings: Stores the path towards review embedding file
    :param path_to_reviews_embedding_matrix: Stores the path towards the matrix that contains all the embedding
    for all the reviews
    :param path_to_item_review_count: Stores the path towards item tensor that stores how many reviews for each item
    """
    _items_id: np.ndarray
    _items_reviews_embedding: pd.DataFrame
    _reviews_embedding_matrix: torch.Tensor
    _num_of_reviews_per_restaurant: torch.Tensor

    def __init__(self, embedder: BERT_model, path_to_items_id: str, path_to_items_review_embeddings: str,
                 path_to_reviews_embedding_matrix: str, path_to_item_review_count: str):
        super().__init__(embedder)
        self._items_id = np.load(path_to_items_id)
        self._items_reviews_embedding = pd.read_csv(path_to_items_review_embeddings)
        self._reviews_embedding_matrix = torch.load(path_to_reviews_embedding_matrix)
        self._num_of_reviews_per_restaurant = torch.load(path_to_item_review_count)

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_review_count: torch.Tensor,
                        item_ids_to_keep: np.ndarray) -> tuple[list, list]:
        """
        This function takes a query and returns a list of business id that is most similar to the query and the top k
        reviews for that item

        :param query: The input information retriever gets
        :param topk_items: Number of items to be returned
        :param topk_reviews: Number of reviews for each item
        :param item_review_count: A pytorch tensor that contains the amount of reviews each item have
        :param item_ids_to_keep: Stores the item id to keep in a numpy array
        :return: Return a tuple with element 0 being a list[str] a list of string containing the most similar item's
        item_id and element 1 being list[list[str]] with Dim 0 has the top k most similar item, Dim 1 has the top k
        reviews for the corresponding item
        """
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._similarity_score_each_review(query_embedding, self._reviews_embedding_matrix)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(similarity_score_review,
                                                                                            item_review_count,
                                                                                            topk_reviews)
        # Finds the indexes of the ids to count
        id_index = self._find_index(item_ids_to_keep)
        mask = torch.full_like(similarity_score_item, False, dtype=torch.bool)
        mask[id_index] = True
        similarity_score_item[~mask] = 0

        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items)
        list_of_item_id = self._get_topk_item_item_id(most_similar_item_index, self._items_reviews_embedding)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review,
                                          self._items_reviews_embedding)

        return list_of_item_id, list_of_review

    def _find_index(self, ids_to_keep: np.array) -> list[int]:
        indices = []

        for id in ids_to_keep:
            indices.append(np.where(self._items_id == id)[0][0])

        return indices

    @staticmethod
    def _similarity_score_each_review(query: torch.Tensor, reviews: torch.Tensor) -> torch.Tensor:
        """
        This function finds and returns a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :param reviews: A matrix of all the review embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """

        # Get the similarity score using matrix multiplication
        similarity_score = torch.matmul(reviews, query)

        return similarity_score

    @staticmethod
    def _get_topk_item_item_id(most_similar_item_index: torch.Tensor, df: pd.DataFrame) -> list[str]:
        """
        Get the most similar item's business id

        :param most_similar_item_index: A tensor containing the top k items with the most
        :return: A list of business id of the most similar items. Beginning from the most
            similar to the least.
        """
        unique_values = df["Item_ID"].unique()
        list_of_item_id = []
        most_similar_item_index = most_similar_item_index.tolist()
        for i in most_similar_item_index:
            list_of_item_id.append(unique_values[i])

        return list_of_item_id

    @staticmethod
    def _get_review(most_similar_item_index: torch.Tensor, index_most_similar_review: torch.Tensor,
                    items_reviews_embedding: pd.DataFrame) -> list[list[str]]:
        """
        Return the most similar reviews for those top k items

        :param most_similar_item_index: A tensor containing the index of the most similar items
        :param index_most_similar_review: A tensor containing the index of all the items(Not just the most similar item)
        :param items_reviews_embedding: A panda object that reads from the review embedding file :return: returns
         a list[list[
        str]] with dim 0 has the top k most similar item, dim 1 has the top k reviews for the corresponding item
        """

        most_similar_item_index = most_similar_item_index.tolist()
        index_most_similar_review = index_most_similar_review.tolist()
        review_list = []
        for i in most_similar_item_index:
            most_similar_review_list = index_most_similar_review[i]
            review_list_item = []
            for j in most_similar_review_list:
                # if not -1, which was padded to make sure the size each row matches with each other
                if j != -1:
                    review_list_item.append(items_reviews_embedding["Review"][j])

            review_list.append(review_list_item)
        return review_list