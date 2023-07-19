import faiss
import numpy as np


class VectorDataBase:
    """
    This class functions as a vector database

    :param database_file_path: Stores the path towards vector database
    :param id_file_path: Stores the path towards id numpy array
    :param review_file_path: Stores the path towards review numpy array
    """
    _storage: faiss.IndexFlat
    _id: np.ndarray
    _review: np.ndarray
    _ntotal: int

    def __init__(self, database_file_path: str, id_file_path: str, review_file_path: str):
        self._storage = faiss.read_index(database_file_path)
        self._id = np.load(id_file_path, allow_pickle=True)
        self._review = np.load(review_file_path, allow_pickle=True)
        self._ntotal = self._storage.ntotal

    def find_similarity_vector(self, query: np.ndarray) -> np.ndarray:
        query = query.reshape(-1, self._storage.d)
        D, I = self._storage.search(query, self._storage.ntotal)
        D = D[0]
        I = I[0]  # For some reason FAISS return a numpy within a numpy that contains all the answer.

        output = [False] * self._ntotal
        for i, index in enumerate(I):
            output[index] = D[i]

        output = np.array(output)

        return output

    def get_id(self):
        return self._id

    def get_review(self):
        return self._review
