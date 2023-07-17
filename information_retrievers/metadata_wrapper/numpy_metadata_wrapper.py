import numpy as np
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from domain_specific_config_loader import DomainSpecificConfigLoader


class NumpyMetadataWrapper(MetadataWrapper):
    """
    Numpy metadata wrapper that is responsible to get item metadata as dictionary.
    """

    _items_metadata: np.ndarray

    def __init__(self) -> None:
        domain_specific_config_loader = DomainSpecificConfigLoader()
        path_to_items_metadata = domain_specific_config_loader.get_path_to_item_metadata()
        self._items_metadata = np.load(path_to_items_metadata)

    def get_item_dict_from_id(self, item_id: str) -> dict[str, str]:
        """
        Return item metadata as a dictionary from item id.

        :param item_id: item id
        :return: item metadata
        """
        num_item = self._items_metadata.shape[0]
        for index in range(num_item):
            item_metadata_dict = self._items_metadata[index]

            if item_metadata_dict['item_id'] == item_id:
                return item_metadata_dict

    def get_item_dict_from_index(self, index: int) -> dict[str, str]:
        """
        Return item metadata as a dictionary from index.

        :param index: index to the item in the metadata
        :return: item metadata
        """
        return self._items_metadata[index]

    def get_num_item(self) -> int:
        """
        Return the number of items.

        :return: number of items
        """
        return self._items_metadata.shape[0]
