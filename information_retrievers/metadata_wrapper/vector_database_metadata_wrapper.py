import numpy
from information_retrievers.metadata_wrapper.metadata_wrapper import MetadataWrapper


class VectorDatabaseMetadataWrapper(MetadataWrapper):

    def __init__(self):
        pass

    def filter(self) -> numpy.ndarray:
        raise NotImplementedError()

    def get_item_dict(self) -> dict[str, str]:
        raise NotImplementedError()