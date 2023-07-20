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
        domain_specific_config_loader = DomainSpecificConfigLoader()
        self._database, review_item_ids, reviews = domain_specific_config_loader.load_vector_database()
        super().__init__(embedder, review_item_ids, reviews)


    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_indices_to_keep: list[int]) -> tuple[list, list]:
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._database.find_similarity_vector(query_embedding)
        similarity_score_review = torch.tensor(similarity_score_review)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(
            similarity_score_review, topk_reviews)
        similarity_score_item = self._filter_item_similarity_score(similarity_score_item, item_indices_to_keep)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items)
        list_of_business_id = self._get_topk_item_id(
            most_similar_item_index, index_most_similar_review)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review)

        return list_of_business_id, list_of_review


