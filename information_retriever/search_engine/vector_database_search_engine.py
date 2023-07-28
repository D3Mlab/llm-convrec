import torch
import numpy as np
from information_retriever.embedder.bert_embedder import BERT_model
from information_retriever.vector_database import VectorDataBase
from information_retriever.search_engine.search_engine import SearchEngine


class VectorDatabaseSearchEngine(SearchEngine):
    """
    Class that is responsible for searching for topk most relevant items using BERT_model.

    :param embedder: BERT_model to embed query
    :param review_item_ids: item ids corresponding to reviews
    :param reviews: reviews of the items
    :param database: FAISS database containing embeddings of the reviews
    """

    _embedder: BERT_model
    _review_item_ids: np.ndarray
    _reviews: np.ndarray
    _database: VectorDataBase

    def __init__(self, embedder: BERT_model, review_item_ids: np.ndarray, reviews: np.ndarray,
                 database: VectorDataBase):
        super().__init__(embedder, review_item_ids, reviews)
        self._database = database

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        Return a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """
        return self._database.find_similarity_vector(query)
