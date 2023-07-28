import torch
import faiss


class VectorDataBase:
    #TODO: maybe change the wording ehre? not very clear?
    """
    This class functions as a vector database

    :param storage: Stores the vector database
    """

    _ntotal: int
    _storage: faiss.Index

    def __init__(self, storage):
        self._storage = storage
        self._ntotal = self._storage.ntotal

    def find_similarity_vector(self, query: torch.Tensor) -> torch.Tensor:
        """
        This function finds the similarity between the query and the vectors in the database

        :param query: query embedding
        :return: The similarity score between the query and each vector in the database in respect to the index.
        """
        query = query.reshape(-1, self._storage.d)
        D, I = self._storage.search(query, self._storage.ntotal)
        D = D[0]
        I = I[0]  # For some reason FAISS return a numpy within a numpy that contains all the answer.

        output = [False] * self._ntotal
        for i, index in enumerate(I):
            output[index] = D[i]

        output = torch.tensor(output)

        return output

