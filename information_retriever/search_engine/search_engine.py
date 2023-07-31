import numpy as np
import torch

from information_retriever.embedder.bert_embedder import BERT_model


class SearchEngine:
    """
    Class that searches for topk most relevant items.
    
    :param embedder: BERT_model to embed query
    """

    _embedder: BERT_model
    _review_item_ids: np.ndarray
    _reviews: np.ndarray

    def __init__(self, embedder: BERT_model, review_item_ids: np.ndarray, reviews: np.ndarray):
        self._embedder = embedder
        self._review_item_ids = review_item_ids
        self._reviews = reviews

    def search_for_topk(self, query: str, topk_items: int, topk_reviews: int,
                        item_indices_to_keep: list[int], unacceptable_similarity_range: float, max_number_similar_items: int) -> tuple[list[list[str]], list[list[list[str]]]]:
        """
        This function takes a query and returns a list of business id that is most similar to the query and the top k
        reviews for that item

        :param query: The input information retriever gets
        :param topk_items: Number of items to be returned
        :param topk_reviews: Number of reviews for each item
        :param item_indices_to_keep: Stores the item id to keep in a list of int
        :param unacceptable_similarity_range: range of similarity scores that would be considered too small to be able to recommend right away
        :param max_number_similar_items: max number of similar items
        :return: Return a tuple with element 0 being a list[list[str]] which is a list of similar items item_id (similar items are items where their similarity score is less than unacceptable similarity range)
        element 1 being list[list[list[str]]] with Dim 0 has the group of similar items, Dim 1 has the similar item, Dim 2 has the top k
        reviews for the corresponding item
        """
        query_embedding = self._embedder.get_tensor_embedding(query)
        similarity_score_review = self._similarity_score_each_review(
            query_embedding)
        similarity_score_item, index_most_similar_review = self._similarity_score_each_item(
            similarity_score_review, topk_reviews)
        similarity_score_item = self._filter_item_similarity_score(similarity_score_item, item_indices_to_keep)
        most_similar_item_index = self._most_similar_item(similarity_score_item, topk_items, unacceptable_similarity_range, max_number_similar_items)
        list_of_item_id = self._get_topk_item_id(most_similar_item_index, index_most_similar_review)
        list_of_review = self._get_review(most_similar_item_index, index_most_similar_review)

        return list_of_item_id, list_of_review

    def _similarity_score_each_review(self, query: torch.Tensor) -> torch.Tensor:
        """
        This function finds and returns a tensor that contains the similarity score for each review

        :param query: A tensor containing the query embedding
        :return: A pytorch tensor that contains the similarity score for each review
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
        prev_id = self._review_item_ids[0]
        item_score = []
        item_index = []
        index_start = 0
        for i in range(self._review_item_ids.size + 1):
            if i == self._review_item_ids.size or self._review_item_ids[i] != prev_id:
                similarity_score_item = similarity_score[index_start:i]
                k_actual = min(i - index_start, k)
                values, index_topk = similarity_score_item.topk(k_actual)
                index_topk += index_start
                item_score.append(values.mean(dim=0))

                if index_topk.size()[0] != k:
                    padding = torch.full((k - index_topk.size()[0],), -1, dtype=torch.int64)
                    index_topk = torch.cat((index_topk, padding), dim=0)
                item_index.append(index_topk)

                index_start = i
            if i != self._review_item_ids.size:
                prev_id = self._review_item_ids[i]

        item_score = torch.stack(item_score)
        item_index = torch.stack(item_index)
        return item_score, item_index

    @staticmethod
    def _most_similar_item(similarity_score_item: torch.Tensor, top_k_items: int, unacceptable_similarity_range: float, max_number_similar_items: int) -> torch.Tensor:
        """
        This function returns the most similar item's index given the item similarity score

        :param similarity_score_item: The similarity score for each item
        :param top_k_items: Number of items to return
        :param unacceptable_similarity_range: range of similarity scores that would be considered too small to be able to recommend right away
        :param max_number_similar_items: max number of similar items 
        :return: The indices of the most similar item, beginning from the most similar to the least similar.
        Indices are grouped by similarity scores, so if values are too close together they will be considered 1 group.
        It returns at most top_k_items *  max_number_similar_items indices.
        """
        values, indices = torch.sort(similarity_score_item, descending = True)
        
        num_non_zero_value = torch.nonzero(values).size(0)
        if num_non_zero_value == 0:
            raise Exception("There are no items that match.")
        
        topk_indices = torch.full((top_k_items, max_number_similar_items), -1)
        topk_indices[0][0] = indices[0]
        
        item_score = values[0]
        
        # A group: a group of indices that would be considred 1 "1 item" so where there similarity score is less than 0.5
        # number of groups = number of top k items
        # max values (or items) per group 
        num_groups = 0
        num_vals = 0
        
        for iteration in range(1, len(values)):            
            if values[iteration] != 0:
                if item_score - values[iteration] <= unacceptable_similarity_range and num_vals <= max_number_similar_items -2:
                    num_vals +=1

                else:
                    item_score = values[iteration]
                    num_groups +=1
                    num_vals = 0
                    
                    # If it exceeds max number of items per group
                    if num_groups == top_k_items:
                        break
                
                topk_indices[num_groups][num_vals] = indices[iteration]

        # sometimes indices are a float for whatever reason
        return topk_indices.to(torch.int64)


    @staticmethod
    def _filter_item_similarity_score(similarity_score_item: torch.Tensor, id_index: list[int]) -> torch.Tensor:
        mask = torch.full_like(similarity_score_item, False, dtype=torch.bool)
        mask[id_index] = True
        similarity_score_item[~mask] = 0
        return similarity_score_item

    def _get_topk_item_id(self, most_similar_item_index: torch.Tensor,
                                   index_most_similar_review: torch.Tensor) -> list[list[str]]:
        """
        Get the most similar item's business id

        :param most_similar_item_index: A tensor containing the top k items with the most
        :return: A list of lists of business id of the most similar items. Beginning from the most
            similar to the least and grouped by items that are considered within an unacceptable range of difference between the similarity scores.
        """

        list_of_id = []

        for i in range(len(most_similar_item_index)):
            id_group = []
            for j in range(len(most_similar_item_index[i])):
                if int(most_similar_item_index[i][j]) != -1:
                    id_group.append(self._review_item_ids[index_most_similar_review[int(most_similar_item_index[i][j])][0]])
            if id_group != []:
                list_of_id.append(id_group)
        return list_of_id

    def _get_review(self, most_similar_item_index: torch.Tensor, index_most_similar_review: torch.Tensor) \
            -> list[list[list[str]]]:
        """
        Return the most similar reviews for those top k items

        :param most_similar_item_index: A tensor containing the index of the most similar items
        :param index_most_similar_review: A tensor containing the index of all the items(Not just the most similar item)
        :return: Returns a list of lists of lists of reviews
        """

        review_list = []
        for i in most_similar_item_index:
            item_group_review_list = []
            for index in i:
                if index != -1:
                    item_review_list = []
                    for j in index_most_similar_review[index]:
                        item_review_list.append(self._reviews[j])
                    item_group_review_list.append(item_review_list)
            if item_group_review_list != []:
                review_list.append(item_group_review_list)
        
        return review_list
