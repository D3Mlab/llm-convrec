from state.state_manager import StateManager
import pandas as pd


class Filter:
    """"
    Responsible to do filtering and return a filtered version of metadata pandas dataframe.
    """

    def filter(self, state_manager: StateManager,
               metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Return a filtered version of metadata pandas dataframe.

        :param state_manager: current state
        :param metadata: items' metadata
        :return: filtered version of metadata pandas dataframe
        """
        raise NotImplementedError()
