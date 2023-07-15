from typing import Any


class Item:
    """
    Class represents an item.
    
    :param id_input: Id of the item
    :param storage: The dictionary that contains the item information
    """
    _id: str
    _mandatory: dict[str, Any]
    _optional: dict[str, Any]

    def __init__(self, id_input: str, name: str, mandatory: dict[str, Any], optional: dict[str, Any] = None):
        if optional is None:
            optional = {}
        self._id = id_input
        self._name = name
        self._mandatory = mandatory
        self._optional = optional

    def get_id(self):
        """
        Get the business id.

        :return: business id stored in the object
        """
        return self._id

    def get_name(self):
        return self._name

    def get_mandatory_data(self):
        return self._mandatory

    def get_optional_data(self) -> dict:
        return self._optional

    def get(self, key):
        return self._mandatory[key]
