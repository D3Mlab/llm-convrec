import re

import pandas as pd
import yaml


class DomainSpecificConfigLoader:

    def __init__(self):
        with open('config.yaml') as f:
            self.system_config = yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def _load_dict_in_cell(data_string):
        pattern = r'([^,]+)\s*=\s*\[([^\]]+)\]'
        matches = re.findall(pattern, data_string)
        data_dict = {key.strip(): [value.strip().removesuffix('"').removeprefix('"').removesuffix('.').strip().lower() for value in
                     re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lst)] for key, lst in matches}
        return data_dict

    def load_domain(self) -> str:
        general_config = pd.read_csv(self.system_config['GENERAL_CONFIG_FILE'], encoding='latin1')
        return general_config.to_dict("records")[0]['domain']

    def load_model(self) -> str:
        general_config = pd.read_csv(self.system_config['GENERAL_CONFIG_FILE'], encoding='latin1')
        return general_config.to_dict("records")[0]['model']

    def load_constraints_categories(self) -> list[dict]:
        constraints_df = pd.read_csv(self.system_config['CONSTRAINTS_CONFIG_FILE'], encoding='latin1')
        return constraints_df.to_dict("records")

    def load_constraints_updater_fewshots(self) -> list[dict]:
        constraints_fewshots_df = pd.read_csv(self.system_config['CONSTRAINTS_UPDATER_FEWSHOTS_FILE'], encoding='latin1')
        constraints_fewshots = [
            {
                'user_input': row['user_input'],
                'old_hard_constraints': self._load_dict_in_cell(row["old_hard_constraints"])
                if isinstance(row["old_hard_constraints"], str) else None,
                'old_soft_constraints': self._load_dict_in_cell(row["old_soft_constraints"])
                if isinstance(row["old_soft_constraints"], str) else None,
                'new_hard_constraints': self._load_dict_in_cell(row["new_hard_constraints"])
                if isinstance(row["new_hard_constraints"], str) else None,
                'new_soft_constraints': self._load_dict_in_cell(row["new_soft_constraints"])
                if isinstance(row["new_soft_constraints"], str) else None,
            }
            for row in constraints_fewshots_df.to_dict("records")
        ]
        return constraints_fewshots
