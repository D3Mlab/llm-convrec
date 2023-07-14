import faiss
import numpy as np
import torch

class VectorDataBase:
    """
    This class functions as a vector database

    :param _storage: Stores the vector database
    """
    _storage: faiss.swigfaiss_avx2
    _id: np.ndarray
    _metadata: np.ndarray
    _review: np.ndarray
    _metadata_storage: np.ndarray
    _ntotal: int

    def __init__(self, database_file_path: str, id_file_path: str, metadata_file_path: str, review_file_path: str, metadata_storage_file_path: str):
        self._storage = faiss.read_index(database_file_path)
        self._id = np.load(id_file_path)
        self._metadata = np.load(metadata_file_path)
        self._review = np.load(review_file_path)
        self._metadata_storage = np.load(metadata_storage_file_path, allow_pickle=True)
        self._ntotal = self._storage.ntotal

    def search_for_index(self, query: np.ndarray, k: int):
        """
        Search the database

        :param query: This is the query vector
        :param k: This is how many items to retrieve
        :return: The indexs of most similar vectors
        """
        # First output stores the distance between query and retrieved vectors
        # I stores the index of retrieved vectors
        _, I = self._storage.search(query, k)

        return I

    def search_for_vector(self, query: np.ndarray, k: int):
        """
        Search the database and return the actual embedding vectors

        :param query: This is the query vector
        :param k: This is how many items to retrieve
        :return: A numpy array containing the actual most similar vectors
        """
        # First output stores the distance between query and retrieved vectors
        # I stores the index of retrieved vectors
        _, I = self._storage.search(query, k)

        list_of_vectors = []
        # Create the np array that contains the most similar vectors
        for index in I[0]:
            list_of_vectors.append(self._storage.reconstruct(int(index)))

        np_of_vectors = np.array(list_of_vectors)

        return np_of_vectors

    def filter_with_id(self, target_id: str) -> np.ndarray:
        """
        This function serves as the filter for id

        :param id: A 1d np array
        :param target_id: A string representing the id you are searching for

        :return: A numpy array with the same shape as id, with index of target_id
        set to True while all other index set to false
        """
        return self._id == target_id

    def filter_with_metadata(self, target: str) -> np.ndarray:
        """
        This function examines the metadata and searches for lists that
        include the target. It then generates a one-dimensional numpy array
        where the index corresponds to each list. If a list contains the target,
        the corresponding index in the array is set to True; otherwise,
        it is set to False.

        :param metadata: A 2d list containing metadata
        :param target: A string representing the target we are filtering for

        :return: A 1d numpy array with True represents this item's metadata
        contains the target and False represent otherwise
        """

        indexes = np.zeros((self._ntotal), dtype=bool)

        items_satisfies_requirement = []

        for i, info in enumerate(self._metadata_storage):
            for key in info.keys():
                if isinstance(info[key], dict):
                    if(target in info[key].values()):
                        items_satisfies_requirement.append(i)
                        break
                else:
                    if(target == info[key]):
                        items_satisfies_requirement.append(i)
                        break
                    if isinstance(info[key], str):
                        if(target in info[key]):
                            items_satisfies_requirement.append(i)
                            break

        for number in items_satisfies_requirement:
            id_filter = self._metadata == number
            indexes = np.logical_or(indexes, id_filter)

        return indexes

    def search_with_filter(self, query: np.ndarray, k: int, target_id: list = None, target_metadata: list = None) -> np.ndarray:
        """
        This function filters the datavase to look for indexs with metadata
        that contains the target we are looking for and items with id we are looking for.

        :param query: The query embedding of shape [1, 768]
        :param k: The number of vectors we want to return
        :param target_id: The target id we are looking for
        :param target_metadata: The metadata we are looking for

        :return: The indexs of the review
        """
        # Create id filter
        id_filter = np.ones((self._ntotal), dtype=bool)
        if(target_id != None):
            # If user did not specify what id they are looking for
            # we are not going to filter out anything
            id_filter = np.zeros((self._ntotal), dtype=bool)
            for id in target_id:
                id_filter_requirement = self.filter_with_id(id)
                id_filter = np.logical_or(id_filter, id_filter_requirement)

        # Create metadata filter
        metadata_filer = np.ones((self._ntotal), dtype=bool)
        if(target_metadata != None):
            # If user did not specify the kind of metadata they are looking for
            # we are not going to filter out anything
            for requirement in target_metadata:
                metadata_filer_requirement = self.filter_with_metadata(requirement)
                metadata_filer = np.logical_and(metadata_filer, metadata_filer_requirement)

        mask = np.logical_and(id_filter, metadata_filer)

        vector_search_num = k
        count = np.count_nonzero(mask == True)

        if(count == 0):
            # If the user specifies a filter that no item can satisfy
            print("""The filter you have entered appears to exclude all available
            options. Please review your filter criteria to ensure that it allows
            for the selection of relevant items.""")
            return None
        if(count < k):
            # If the user ask for more retrieved item than there is
            print("""The number of items you want to retrieve is more than number of items that satisfies
            your requirements.""")
            return None

        # Actually searching
        _, I = self._storage.search(query, self._storage.ntotal)
        filtered_indices = I[0][mask[I[0]]] # Indices in this variable satisfies the filtering requirements
        selected_index = filtered_indices[:k] #only return the top k relavance indices
        return selected_index

    def find_similarity_vector(self, query: np.ndarray) -> np.ndarray:
        query = query.reshape(-1, self._storage.d)
        D, I = self._storage.search(query, self._storage.ntotal)
        D = D[0]
        I = I[0] #For some reason FAISS return a numpy within a numpy that contains all the answer.

        output = [False] * self._ntotal
        for i, index in enumerate(I):
            output[index] = D[i]

        output = np.array(output)
        return output

    def get_database_size(self):
        """
        This function finds how many vectors this database is storing

        :return: The size of the database
        """
        return self._ntotal

    def get_vector_size(self):
        """
        This function finds the size of the vector this database is storing

        :return: The size of the vector
        """
        return self._storage.d