import numpy
import torch
import pandas as pd
from information_retrievers.embedder.bert_embedder import BERT_model
from information_retrievers.search_engine import SearchEngine

class VectorDatabaseSearchEngine(SearchEngine):
    """
    Class that seaes for topk most relevant items using vector database.
    
    :param embedder: BERT_model to embed query
    #TODO add params necessary for vector database
    """
    
    def __init__(self, embedder: BERT_model, path_to_items_meta_data: str, path_to_items_review_embeddings: str,
                 path_to_reviews_embedding_matrix: str, path_to_item_review_count: str):
        super.__init__(embedder)
        self._restaurants_meta_data = pd.read_csv(path_to_items_meta_data)
        self._restaurants_reviews_embedding = pd.read_csv(path_to_items_review_embeddings)
        self._reviews_embedding_matrix = self._load_matrix(path_to_reviews_embedding_matrix)
        self._num_of_reviews_per_restaurant = self._load_item(path_to_item_review_count)

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
        #TODO implement this function
        raise NotImplementedError()
    
    #TODO implement helper functions
    
    def _load_matrix(self, matrix_file_path: str):
        """
        Load the review embedding tensor 

        :param matrix_file_path: path to the pt file that matrix is stored
        :return: torch tensor that contains review embedding matrix
        """
        matrix_tensor = torch.load(matrix_file_path).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return matrix_tensor

    def _load_item(self, item_review_count_file_path: str):
        """
        Load the tensor that contains how many review each item has

        :param file_path: path to the pt file that item review count is stored
        :return: torch tensor that contains how many review each item has
        """
        item_tensor = torch.load(item_review_count_file_path)
        return item_tensor