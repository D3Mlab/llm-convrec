import torch
import numpy as np
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.vector_database import VectorDataBase
from information_retrievers.search_engine.search_engine import SearchEngine
from domain_specific_config_loader import DomainSpecificConfigLoader


class VectorDatabaseSearchEngine(SearchEngine):
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.

    :param embedder: BERT_model to embed query
    """

    _embedder: BERT_model
    _database: VectorDataBase
    _item_review_count: torch.Tensor
    _items_id: np.ndarray

    def __init__(self, embedder: BERT_model):
        super().__init__(embedder)
        domain_specific_config_loader = DomainSpecificConfigLoader()
        self._database = domain_specific_config_loader.load_vector_database()

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_ids_to_keep: np.ndarray) -> tuple[list, list]:
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._database.find_similarity_vector(query_embedding)
        similarity_score_review = torch.tensor(similarity_score_review)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(
            similarity_score_review, topk_reviews)
        id_index = self._find_index(item_ids_to_keep)
        similarity_score_item = self._filter_item_similarity_score(similarity_score_item, id_index)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items)
        list_of_business_id = self._get_topk_item_business_id(
            most_similar_item_index, index_most_similar_review)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review)

        return list_of_business_id, list_of_review

    def _get_topk_item_business_id(self, most_similar_item_index: torch.Tensor,
                                   index_most_similar_review: torch.Tensor) -> list[str]:
        """
        Get the most similar item's business id

        :param most_similar_item_index: A tensor containing the top k items with the most
        :return: A list of business id of the most similar items. Beginning from the most
            similar to the least.
        """

        list_of_id = []

        for i in range(len(most_similar_item_index)):
            list_of_id.append(self._database.get_id()[index_most_similar_review[most_similar_item_index[i]][0]])

        return list_of_id

    def _get_review(self, most_similar_item_index: torch.Tensor, index_most_similar_review: torch.Tensor) \
            -> list[list[str]]:
        """
        Return the most similar reviews for those top k items

        :param most_similar_item_index: A tensor containing the index of the most similar items
        :param index_most_similar_review: A tensor containing the index of all the items(Not just the most similar item)
        :return: Returns a list of lists of reviews
        """

        review_list = []
        for i in most_similar_item_index:
            item_review_list = []
            for j in index_most_similar_review[i]:
                item_review_list.append(self._database.get_review()[j])

            review_list.append(item_review_list)
        return review_list
