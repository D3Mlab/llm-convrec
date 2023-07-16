class MetadataWrapper:
    """
    Metadata wrapper that is responsible to get item metadata as dictionary.
    """

    def get_item_dict_from_id(self, item_id: str) -> dict[str, str]:
        """
        Return item metadata as a dictionary from item id.

        :param item_id: item id
        :return: item metadata
        """
        raise NotImplementedError()

    def get_item_dict_from_index(self, index: int) -> dict[str, str]:
        """
        Return item metadata as a dictionary from index.

        :param index: index to the item in the metadata
        :return: item metadata
        """
        raise NotImplementedError()

    def get_num_item(self) -> int:
        """
        Return the number of items.

        :return: number of items
        """
        raise NotImplementedError()

