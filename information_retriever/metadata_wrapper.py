import pandas as pd
from typing import Any


class MetadataWrapper:
    """
    Metadata wrapper that is responsible to get item metadata as dictionary.

    :param items_metadata: metadata of all items
    """

    items_metadata: pd.DataFrame

    def __init__(self, items_metadata: pd.DataFrame):
        self.items_metadata = items_metadata

    def get_item_dict_from_id(self, item_id: str) -> dict[str, Any]:
        """
        Return item metadata as a dictionary from item id.

        :param item_id: item id
        :return: item metadata
        """
        item_metadata = self.items_metadata.loc[self.items_metadata['item_id'] == item_id].iloc[0]
        return item_metadata.to_dict()

    def get_item_dict_from_index(self, index: int) -> dict[str, Any]:
        """
        Return item metadata as a dictionary from index.

        :param index: index to the item in the metadata
        :return: item metadata
        """
        return self.items_metadata.loc[index].to_dict()

    def get_metadata(self) -> pd.DataFrame:
        """
        Return the metadata dataframe.

        :return: metadata dataframe
        """
        return self.items_metadata.copy()
