import numpy
import torch
from information_retrievers.embedder.bert_embedder import BERT_model

class SearchEngine:
    """
    Class that seaes for topk most relevant items.
    
    :param embedder: BERT_model to embed query
    """
    
    def __init__(self, embedder: BERT_model):
        self._embedder = embedder

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_review_count: torch.Tensor, 
                        item_ids_to_keep: numpy.ndarray) -> tuple[list[str], list[list[str]]]:
        """
        Takes a query and returns a list of item id that is most similar to the query and the top k
        reviews for that item

        :param query: query text
        :param topk_items: number of items to be returned
        :param topk_reviews: number of reviews for each item
        :param item_review_count: number of reviews for each item 
        :return: a tuple where the first element is a list of most relevant item's id 
        and the second element is a list of top k reviews for each most relevant item
        """
        raise NotImplementedError()
    
    #TODO add helper functions (implement shared helpers here and non shared helpers in child classes,
    # for non shared helpers that need different parameters, don't put the definitions here, 
    # just put them in child classes)