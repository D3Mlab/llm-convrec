from typing import Any


class Item:
    """
    Class represents an item.
    
    :param id: Id of the item
    :param storage: The dictionary that contains the item information
    """
    _id: str
    _storage: dict[Any]

    def __init__(self, id_input: str, storage: dict[Any]):
        self._id = id_input
        self._storage = storage

    def get_id(self):
        """
        Get the business id.

        :return: business id stored in the object
        """
        return self._id

    def get_storage(self):
        return self._storage

    def get(self, key):
        return self._storage[key]
