import numpy


class MetadataWrapper:

    def filter(self) -> numpy.ndarray:
        raise NotImplementedError()

    def get_item_dict(self) -> dict[str, str]:
        raise NotImplementedError()
