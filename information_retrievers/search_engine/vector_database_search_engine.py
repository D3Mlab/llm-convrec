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
    _review_item_ids: np.ndarray
    _reviews: np.ndarray
    _database: VectorDataBase

    def __init__(self, embedder: BERT_model):
        domain_specific_config_loader = DomainSpecificConfigLoader()
        review_item_ids, reviews, self._database =\
            domain_specific_config_loader.load_data_for_vector_database_search_engine()
        super().__init__(embedder, review_item_ids, reviews)

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        Return a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """
        similarity_score_review = self._database.find_similarity_vector(query)
        similarity_score_review = torch.tensor(similarity_score_review)
        return similarity_score_review


