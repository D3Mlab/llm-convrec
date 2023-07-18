from state.state_manager import StateManager
import pandas as pd


class Filter:

    def filter(self, state_manager: StateManager,
               filtered_metadata: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
