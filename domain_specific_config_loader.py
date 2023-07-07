import re

import pandas as pd
import yaml


class DomainSpecificConfigLoader:

    def __init__(self):
        with open('system_config.yaml') as f:
            self.system_config = yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def _load_dict_in_cell(data_string):
        pattern = r'([^,]+)\s*=\s*\[([^\]]+)\]'
        matches = re.findall(pattern, data_string)
        data_dict = {key.strip(): [value.strip().removesuffix('"').removeprefix('"').removesuffix('.').strip().lower() for value in
                     re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lst)] for key, lst in matches}
        return data_dict

    def load_domain(self) -> str:
        path_to_csv = self.load_domain_specific_config()['DOMAIN']
        general_config = pd.read_csv(path_to_csv, encoding='latin1')
        return general_config.to_dict("records")[0]['domain']

    def load_model(self) -> str:
        return self.system_config['MODEL']

    def load_constraints_categories(self) -> list[dict]:
        path_to_csv = self.load_domain_specific_config()[
            'CONSTRAINTS_CATEGORIES']
        constraints_df = pd.read_csv(path_to_csv, encoding='latin1')
        return constraints_df.to_dict("records")

    def load_constraints_updater_fewshots(self) -> list[dict]:
        path_to_csv = self.load_domain_specific_config()[
            'CONSTRAINTS_UPDATER_FEWSHOTS']
        constraints_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
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

    def load_domain_specific_config(self):
        path_to_domain = self.system_config['PATH_TO_DOMAIN_CONFIGS']

        with open(f'{path_to_domain}/domain_specific_config.yaml') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
