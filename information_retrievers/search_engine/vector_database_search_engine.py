import torch
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.search_engine import SearchEngine
from information_retrievers.ir.vector_database import VectorDataBase

class VectorDatabaseSearchEngine(SearchEngine):
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
        list_of_business_id = self._get_topk_item_business_id(most_similar_item_index)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review)

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

    def _get_topk_item_business_id(self, most_similar_item_index: torch.Tensor) -> list[str]:
        """
        Get the most similar item's business id

        :param most_similar_item_index: A tensor containing the top k items with the most
        :return: A list of business id of the most similar items. Beginning from the most
            similar to the least.
        """

        list_of_id = []

        for i in most_similar_item_index:
            list_of_id.append(self._database._metadata_storage[i]["item_id"])

        return list_of_id

    def _get_review(self, most_similar_item_index: torch.Tensor, index_most_similar_review: torch.Tensor) -> list[str]:
        """
        Return the most similar reviews for those top k items

        :param most_similar_item_index: A tensor containing the index of the most similar items
        :param index_most_similar_review: A tensor containing the index of all the items(Not just the most similar item)
        :param items_reviews_embedding: A panda object that reads from the review embedding file :return: returns a list[list[
        str]] with dim 0 has the top k most similar item, dim 1 has the top k reviews for the corresponding item
        """

        review_list = []
        for i in most_similar_item_index:
            item_review_list = []
            for j in index_most_similar_review[i]:
                item_review_list.append(self._database._review[j])

            review_list.append(item_review_list)
        return review_list
