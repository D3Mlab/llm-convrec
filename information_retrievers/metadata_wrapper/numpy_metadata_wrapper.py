import numpy as np
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper
from information_retrievers.checker.checker import Checker
from state.state_manager import StateManager

class NumpyMetadataWrapper(MetadataWrapper):
    """
    Numpy metadata wrapper that is responsible to do filter and get item metadata as dictionary.

    :param path_to_items_metadata: path to items metadata file
    """

    _items_metadata: np.ndarray

    def __init__(self, path_to_items_metadata: str):
        self._items_metadata = np.load(path_to_items_metadata)

    def filter(self, checkers: list[Checker], state_manager: StateManager) -> np.ndarray:
        """
        Return a numpy array that has item ids that must be kept.
        """
        num_item = self._items_metadata.shape[0]
        item_id_to_keep = []
        for index in range(num_item):
            item_metadata_dict = self._items_metadata[index]

            if self.should_keep_item(checkers, state_manager, item_metadata_dict):
                item_id_to_keep.append(item_metadata_dict['item_id'])

        return np.ndarray(item_id_to_keep)

    def get_item_dict(self, item_id: str) -> dict[str, str]:
        """
        Return item metadata as a dictionary from item id.
        """
        num_item = self._items_metadata.shape[0]
        for index in range(num_item):
            item_metadata_dict = self._items_metadata[index]

            if item_metadata_dict['item_id'] == item_id:
                return item_metadata_dict
