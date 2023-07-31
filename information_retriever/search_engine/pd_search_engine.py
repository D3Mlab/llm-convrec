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

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        This function finds and returns a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :return: A pytorch tensor that contains the similarity score for each review
        """
        similarity_score = torch.matmul(self._reviews_embedding_matrix, query)
        return similarity_score
