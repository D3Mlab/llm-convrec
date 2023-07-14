from information_retrievers.ir.embedder import BERT_model
from information_retrievers.ir.vector_database import VectorDataBase
import pandas as pd
import torch

class NeuralSearchEngine:
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.

    :param embedder: BERT_model to embed query
    """
    _embedder: BERT_model
    _database: VectorDataBase

    def __init__(self, embedder: BERT_model, database: VectorDataBase):
        self._embedder = embedder
        self._database = database

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        items_reviews_embedding: pd.DataFrame,
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
        similarity_score_review = self._similarity_score_each_review(query_embedding)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(similarity_score_review,
                                                                                            item_review_count,
                                                                                            topk_reviews)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items)
        list_of_business_id = self._get_topk_item_business_id(most_similar_item_index, items_reviews_embedding)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review, items_reviews_embedding)

        return list_of_business_id, list_of_review

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        This function finds and returns a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :param reviews: A matrix of all the review embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """

        # Get the similarity score using vector database
        similarity_score = self._database.find_similarity_vector(query.numpy())
        similarity_score = torch.tensor(similarity_score)

        return similarity_score

    @staticmethod
    def _similarity_score_each_item(similarity_score: torch.Tensor, item_review_count: torch.Tensor,
                                    k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function finds and returns a tensor that contains the similarity score for each item

        :param similarity_score: A tensor of similarity score between each review and the query
        :param item_review_count: A tensor containing how many review each item has
        :param k: A number that tells the number of most similar tensors to look at when doing late fusion(k)
        :return: Returning a tuple with element 0 being a tensor that contains the similarity score for each item
                and element 1 being a tensor that contains the index of top k reviews for each item
        """

        index = 0
        # size records how many items are in the matrix
        size = item_review_count.size(0)

        item_score = []
        item_index = []

        for i in range(size):
            # Mask out the review scores related to one item
            similarity_score_item = similarity_score[index:index + item_review_count[i]]

            # Get the top k review scores or all review scores if the number of reviews is less than k
            k_actual = min(item_review_count[i], k)
            values, index_topk = similarity_score_item.topk(k_actual)

            index_topk += index

            # Get the item score by finding the mean of all the review scores
            item_score.append(values.mean(dim=0))

            # if size is smaller than k, pad with -1
            if index_topk.size()[0] != k:
                padding = torch.full((k - index_topk.size()[0],), -1, dtype=torch.int64)
                index_topk = torch.cat((index_topk, padding), dim=0)
            item_index.append(index_topk)

            index += item_review_count[i]

        item_score = torch.stack(item_score)
        item_index = torch.stack(item_index)
        return item_score, item_index

    @staticmethod
    def _most_similar_item(similarity_score_item: torch.Tensor, top_k_items: int) -> torch.Tensor:
        """
        This function returns the most similar item's index given the item similarity score

        :param similarity_score_item: The similarity score for each item
        :param top_k_items: Number of items to return
        :return: The indices of the most similar item, beginning from the most similar to the least similar.
        It returns at most top_k_items indices.
        """

        values, indices = similarity_score_item.topk(top_k_items)
        num_non_zero_value = torch.nonzero(values).size(0)
        if num_non_zero_value == 0:
            raise Exception("There is no restaurants near that location.")
        # remove restaurant that has score of 0
        elif num_non_zero_value < top_k_items:
            # remove restaurant that has score of 0
            num_items_to_remove = top_k_items - num_non_zero_value
            indices = indices[:-num_items_to_remove]
        return indices

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
