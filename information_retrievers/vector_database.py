import numpy as np


class VectorDataBase:
    """
    This class functions as a vector database

    :param storage: Stores the vector database
    """

    _ntotal: int

    def __init__(self, storage):
        self._storage = storage
        self._ntotal = self._storage.ntotal

    def find_similarity_vector(self, query: np.ndarray) -> np.ndarray:
        """
        #TODO
        """
        query = query.reshape(-1, self._storage.d)
        D, I = self._storage.search(query, self._storage.ntotal)
        D = D[0]
        I = I[0]  # For some reason FAISS return a numpy within a numpy that contains all the answer.

        output = [False] * self._ntotal
        for i, index in enumerate(I):
            output[index] = D[i]

        output = np.array(output)

        return output

