import numpy
import torch
import pandas as pd
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.search_engine import SearchEngine

class PDSearchEngine(SearchEngine):
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.
    
    :param embedder: BERT_model to embed query
    """

    def __init__(self, embedder: BERT_model):
        self._embedder = embedder

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        items_reviews_embedding: pd.DataFrame, matrix: torch.Tensor,
                        item_review_count: torch.Tensor) -> tuple[list, list]:
        """
        This function takes a query and returns a list of business id that is most similar to the query and the top k
        reviews for that item

        :param query: The input information retriever gets
        :param topk_items: Number of items to be returned
        :param topk_reviews: Number of reviews for each item 
        :param items_reviews_embedding: Gets the panda object that read from review embedding sorted file
        :param matrix: A pytorch tensor that contains the matrix for review embedding
        :param item_review_count: A pytorch tensor that contains the amount of reviews each item have
        :return: Return a tuple with element 0 being a list[str] a list of string containing the most similar item's
        business_id and element 1 being list[list[str]] with Dim 0 has the top k most similar item, Dim 1 has the top k
        reviews for the corresponding item
        """
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._similarity_score_each_review(query_embedding, matrix)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(similarity_score_review,
                                                                                            item_review_count,
                                                                                            topk_reviews)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items)
        list_of_business_id = self._get_topk_item_business_id(most_similar_item_index, items_reviews_embedding)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review, items_reviews_embedding)

        return list_of_business_id, list_of_review

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
    def _get_topk_item_business_id(most_similar_item_index: torch.Tensor, df: pd.DataFrame) -> list[str]:
        """
        Get the most similar item's business id

        :param most_similar_item_index: A tensor containing the top k items with the most
        :return: A list of business id of the most similar items. Beginning from the most
            similar to the least.
        """
        unique_values = df["Business_ID"].unique()
        list_of_business_id = []
        most_similar_item_index = most_similar_item_index.tolist()
        for i in most_similar_item_index:
            list_of_business_id.append(unique_values[i])

        return list_of_business_id

    @staticmethod
    def _get_review(most_similar_item_index: torch.Tensor, index_most_similar_review: torch.Tensor,
                    items_reviews_embedding: pd.DataFrame) -> list[str]:
        """
        Return the most similar reviews for those top k items

        :param most_similar_item_index: A tensor containing the index of the most similar items
        :param index_most_similar_review: A tensor containing the index of all the items(Not just the most similar item)
        :param items_reviews_embedding: A panda object that reads from the review embedding file :return: returns a list[list[
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
