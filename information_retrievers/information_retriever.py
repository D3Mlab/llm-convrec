from information_retrievers.item.recommended_item import RecommendedItem
import torch
import pandas as pd

class InformationRetriever:
    """
    Abstract class of information retriever that is responsible of retrieving restaurants based on query.
    """
    def get_best_matching_items(self, query: str, topk_items: int, topk_reviews: int,
                                filtered_embedding_matrix: torch.Tensor) -> list[RecommendedItem]:
        raise NotImplementedError
    
    def get_best_matching_reviews_of_item(self, query: str, item_name: list[str],
                                          num_of_reviews_to_return: int,
                                          filtered_restaurants_review_embeddings: pd.DataFrame,
                                          filtered_embedding_matrix: torch.Tensor,
                                          filtered_num_of_reviews_per_restaurant: torch.Tensor) -> list[list[str]]:
        raise NotImplementedError
    
    
