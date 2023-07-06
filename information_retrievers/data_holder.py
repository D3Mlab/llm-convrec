import pandas as pd
import torch


class DataHolder:
    """
    Class that holds all data needed for filters and information retrieval.

    :param path_to_restaurants_meta_data: path to the file that has metadata of restaurants
    :param path_to_restaurants_reviews_embedding: path to the file that has reviews, review embeddings,
    and business ids of restaurants
    :param path_to_reviews_embedding_matrix: path to the file that has matrix of review embeddings
    where a row represents an embedding
    :param path_to_num_of_reviews_per_restaurant: path to the file that has number of reviews per restaurant
    """

    _path_to_restaurants_meta_data: str
    _path_to_restaurants_reviews_embedding: str
    _path_to_reviews_embedding_matrix: str
    _path_to_num_of_reviews_per_restaurant: str

    def __init__(self, path_to_restaurants_meta_data: str, path_to_restaurants_reviews_embedding: str,
                 path_to_reviews_embedding_matrix: str, path_to_num_of_reviews_per_restaurant: str) -> None:
        self._restaurants_meta_data = pd.read_csv(path_to_restaurants_meta_data)
        self._restaurants_reviews_embedding = pd.read_csv(path_to_restaurants_reviews_embedding)
        self._reviews_embedding_matrix = self._load_matrix(path_to_reviews_embedding_matrix)
        self._num_of_reviews_per_restaurant = self._load_item(path_to_num_of_reviews_per_restaurant)

    def get_item_metadata(self) -> pd.DataFrame:
        return self._restaurants_meta_data

    def get_item_reviews_embedding(self) -> pd.DataFrame:
        return self._restaurants_reviews_embedding

    def get_item_embedding_matrix(self) -> torch.Tensor:
        return self._reviews_embedding_matrix

    def get_num_of_reviews_per_item(self) -> torch.Tensor:
        return self._num_of_reviews_per_restaurant

    def _load_matrix(self, file_path: str):
        """
        Load the review embedding tensor into a variable in this class

        :param file_path: path to that stored pt file
        :return: A tensor containing the information obtained from the file path
        """
        matrix_tensor = torch.load(file_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return matrix_tensor

    def _load_item(self, file_path: str):
        """
        Load the tensor that contains how many review each restaurant has

        :param file_path: path to that stored pt file
        :return: A pytorch tensor containing the information obtained from the file path
        """
        item_tensor = torch.load(file_path)
        return item_tensor
