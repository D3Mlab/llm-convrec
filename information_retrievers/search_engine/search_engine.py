import numpy as np
import torch
from information_retrievers.embedder.bert_embedder import BERT_model
from domain_specific_config_loader import DomainSpecificConfigLoader


class SearchEngine:
    """
    Class that searches for topk most relevant items.
    
    :param embedder: BERT_model to embed query
    """

    _embedder: BERT_model
    _item_review_count: torch.Tensor
    _items_id: np.ndarray

    def __init__(self, embedder: BERT_model):
        self._embedder = embedder
        domain_specific_config_loader = DomainSpecificConfigLoader()
        self._item_review_count = domain_specific_config_loader.load_item_review_count()
        self._items_id = domain_specific_config_loader.load_item_id()

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_ids_to_keep: np.ndarray) -> tuple[list, list]:
        """
        Takes a query and returns a list of item id that is most similar to the query and the top k
        reviews for that item

        :param query: query text
        :param topk_items: number of items to be returned
        :param topk_reviews: number of reviews for each item
        :param item_ids_to_keep: The numpy array containing item ids to keep
        :return: a tuple where the first element is a list of most relevant item's id 
        and the second element is a list of top k reviews for each most relevant item
        """
        raise NotImplementedError()

    def _similarity_score_each_item(self, similarity_score: torch.Tensor,
                                    k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function finds and returns a tensor that contains the similarity score for each item

        :param similarity_score: A tensor of similarity score between each review and the query
        :param k: A number that tells the number of most similar tensors to look at when doing late fusion(k)
        :return: Returning a tuple with element 0 being a tensor that contains the similarity score for each item
                and element 1 being a tensor that contains the index of top k reviews for each item
        """

        index = 0
        # size records how many items are in the matrix
        size = self._item_review_count.size(0)

        item_score = []
        item_index = []

        for i in range(size):
            # Mask out the review scores related to one item
            similarity_score_item = similarity_score[index:index + self._item_review_count[i]]

            # Get the top k review scores or all review scores if the number of reviews is less than k
            k_actual = min(self._item_review_count[i], k)

            values, index_topk = similarity_score_item.topk(k_actual)

            index_topk += index

            # Get the item score by finding the mean of all the review scores
            item_score.append(values.mean(dim=0))

            # if size is smaller than k, pad with -1
            if index_topk.size()[0] != k:
                padding = torch.full((k - index_topk.size()[0],), -1, dtype=torch.int64)
                index_topk = torch.cat((index_topk, padding), dim=0)
            item_index.append(index_topk)

            index += self._item_review_count[i]

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

    def _find_index(self, ids_to_keep: np.array) -> list[int]:
        indices = []

        for item_id in ids_to_keep:
            indices.append(np.where(self._items_id == item_id)[0][0])

        return indices

    @staticmethod
    def _filter_item_similarity_score(similarity_score_item, id_index):
        mask = torch.full_like(similarity_score_item, False, dtype=torch.bool)
        mask[id_index] = True
        similarity_score_item[~mask] = 0
        return similarity_score_item
