import pandas as pd
from typing import Any
from domain_specific_config_loader import DomainSpecificConfigLoader


class MetadataWrapper:
    """
    Metadata wrapper that is responsible to get item metadata as dictionary.
    """

    _items_metadata: pd.DataFrame

    def __init__(self) -> None:
        domain_specific_config_loader = DomainSpecificConfigLoader()
        path_to_items_metadata = domain_specific_config_loader.get_path_to_item_metadata()
        self._items_metadata = pd.read_json(path_to_items_metadata, orient='records', lines=True)

    def get_item_dict_from_id(self, item_id: str) -> dict[str, Any]:
        """
        Return item metadata as a dictionary from item id.

        :param item_id: item id
        :return: item metadata
        """
        item_metadata = self._items_metadata.loc[self._items_metadata['item_id'] == item_id].iloc[0]
        return item_metadata.to_dict()

    def get_item_dict_from_index(self, index: int) -> dict[str, Any]:
        """
        Return item metadata as a dictionary from index.

        :param index: index to the item in the metadata
        :return: item metadata
        """
        return self._items_metadata.loc[index]

    def get_num_item(self) -> int:
        """
        Return the number of items.

        :return: number of items
        """
        return self._items_metadata.shape[0]

    def get_metadata(self) -> pd.DataFrame:
        """
        Return the metadata dataframe.

        :return: metadata dataframe
        """
        return self._items_metadata