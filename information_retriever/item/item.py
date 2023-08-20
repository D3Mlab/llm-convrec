from typing import Any


class Item:
    """
    Class represents an item.
    
    :param id_input: unique id of the item
    :param mandatory: The dictionary that contains the mandatory item information
    :param optional: The dictionary that contains the optional item information
    :param images: image urls corresponding to the item
    """
    _id: str
    _name: str
    _mandatory: dict[str, Any]
    _optional: dict[str, Any]
    _images: list[str]

    def __init__(self, id_input: str, name: str, mandatory: dict[str, Any], optional: dict[str, Any] = None,
                 images: list[str] = None):
        if optional is None:
            optional = {}
        if images is None:
            images = []
        self._id = id_input
        self._name = name
        self._mandatory = mandatory
        self._optional = optional
        self._images = images

    def get_id(self) -> str:
        """
        Get the item id.

        :return: item id stored in the object
        """
        return self._id

    def get_name(self) -> str:
        """
        Get name of the item

        :return: name of the item
        """
        return self._name

    def get_mandatory_data(self) -> dict[str, Any]:
        """
        Get mandatory data of the item

        :return: mandatory data of the item
        """
        return self._mandatory

    def get_optional_data(self) -> dict[str, Any]:
        """
        Get optional data of the item

        :return: optional data of the item
        """
        return self._optional
    
    def get_data(self) -> dict[str, Any]:
        """
        Get union of optional and mandatory data of the item

        :return: union of optional and mandatory data of the item
        """
        return self._mandatory | self._optional

    def get(self, key: str) -> Any:
        """
        Get value from mandatory data based on the given key

        :param key: key in the data
        :return: value from mandatory data corresponding to given key
        """
        return self._mandatory[key]

    def get_images(self) -> list[str]:
        """
        Get image urls corresponding to this item.

        :return: image urls corresponding to this item
        """
        return self._images
