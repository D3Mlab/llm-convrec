import re
import pandas as pd
import yaml
from information_retrievers.checker.checker import Checker
from information_retrievers.checker.exact_word_matching_checker import ExactWordMatchingChecker
from information_retrievers.checker.item_checker import ItemChecker
from information_retrievers.checker.location_checker import LocationChecker
from information_retrievers.checker.value_range_checker import ValueRangeChecker
from information_retrievers.checker.word_in_checker import WordInChecker
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper


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
        return self.load_domain_specific_config()['DOMAIN']

    def load_constraints_categories(self) -> list[dict]:
        constraints_category_filename = self.load_domain_specific_config()[
            'CONSTRAINTS_CATEGORIES']

        path_to_csv = f'{self._get_path_to_domain()}/{constraints_category_filename}'
        constraints_df = pd.read_csv(path_to_csv, encoding='latin1')
        return constraints_df.to_dict("records")

    def load_accepted_items_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ACCEPTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        accepted_items_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        accepted_items_fewshots = [
            {
                'user_input': row["user_input"],
                'all_mentioned_items': list(map(lambda x: x.strip(), row['all_mentioned_items'].split(',')))
                if isinstance(row['all_mentioned_items'], str) else [],
                'recently_mentioned_items': list(map(lambda x: x.strip(), row['recently_mentioned_items'].split(',')))
                if isinstance(row['recently_mentioned_items'], str) else [],
                'accepted_items': list(map(lambda x: x.strip(), row['accepted_items'].split(',')))
                if isinstance(row['accepted_items'], str) else [],
            }
            for row in accepted_items_fewshots_df.to_dict("records")
        ]

        return accepted_items_fewshots

    def load_rejected_items_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['REJECTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        rejected_items_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')

        rejected_items_fewshots = [
            {
                'user_input': row["user_input"],
                'all_mentioned_items': list(map(lambda x: x.strip(), row['all_mentioned_items'].split(',')))
                if isinstance(row['all_mentioned_items'], str) else [],
                'recently_mentioned_items': list(map(lambda x: x.strip(), row['recently_mentioned_items'].split(',')))
                if isinstance(row['recently_mentioned_items'], str) else [],
                'rejected_items': list(map(lambda x: x.strip(), row['rejected_items'].split(',')))
                if isinstance(row['rejected_items'], str) else [],
            }
            for row in rejected_items_fewshots_df.to_dict("records")
        ]
        return rejected_items_fewshots

    def load_current_items_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['CURRENT_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        current_items_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        current_items_fewshots = [
            {
                'user_input': row["user_input"],
                'response': row["response"],
            }
            for row in current_items_fewshots_df.to_dict("records")
        ]
        return current_items_fewshots

    def load_constraints_updater_fewshots(self) -> list[dict]:
        constraints_updater_fewshot_filename = self.load_domain_specific_config()[
            'CONSTRAINTS_UPDATER_FEWSHOTS']
        path_to_csv = f'{self._get_path_to_domain()}/{constraints_updater_fewshot_filename}'
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

    def load_answer_extract_category_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ANSWER_EXTRACT_CATEGORY_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        answer_extract_category_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        answer_extract_category_fewshots = [
            {
                'input': row["input"],
                'output': row["output"],
            }
            for row in answer_extract_category_fewshots_df.to_dict("records")
        ]
        return answer_extract_category_fewshots

    def load_answer_ir_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ANSWER_IR_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        answer_ir_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        answer_ir_fewshots = [
            {
                'question': row["question"],
                'information': [row["information"]],
                'answer': row["answer"],
            }
            for row in answer_ir_fewshots_df.to_dict("records")
        ]
        return answer_ir_fewshots

    def load_answer_separate_questions_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ANSWER_SEPARATE_QUESTIONS_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        answer_separate_questions_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        answer_separate_questions_fewshots = [
            {
                'question': row["question"],
                'individual_questions': row["individual_questions"],
            }
            for row in answer_separate_questions_fewshots_df.to_dict("records")
        ]
        return answer_separate_questions_fewshots

    def load_answer_verify_metadata_resp_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ANSWER_VERIFY_METADATA_RESP_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        answer_verify_metadata_resp_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        answer_verify_metadata_resp_fewshots = [
            {
                'question': row["question"],
                'answer': row["answer"],
                'response': row["response"],
            }
            for row in answer_verify_metadata_resp_fewshots_df.to_dict("records")
        ]
        return answer_verify_metadata_resp_fewshots

    def load_domain_specific_config(self):
        path_to_domain = self._get_path_to_domain()

        with open(f'{path_to_domain}/domain_specific_config.yaml') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _get_path_to_domain(self):
        return self.system_config['PATH_TO_DOMAIN_CONFIGS']

    def load_inquire_classification_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['INQUIRE_CLASSIFICATION_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        inquire_classification_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        inquire_classification_fewshots = [
            {
                'input': row["User input"],
                'response': row["Response"],
            }
            for row in inquire_classification_fewshots_df.to_dict("records")
        ]
        return inquire_classification_fewshots
    
    def load_accept_classification_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['ACCEPT_CLASSIFICATION_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        accept_classification_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        accept_classification_fewshots = [
            {
                'input': row["User input"],
                'response': row["Response"],
            }
            for row in accept_classification_fewshots_df.to_dict("records")
        ]
        return accept_classification_fewshots
    
    def load_reject_classification_fewshots(self) -> list[dict]:
        filename = self.load_domain_specific_config()['REJECT_CLASSIFICATION_FEWSHOTS_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        reject_classification_fewshots_df = pd.read_csv(path_to_csv, encoding='latin1')
        reject_classification_fewshots = [
            {
                'input': row["User input"],
                'response': row["Response"],
            }
            for row in reject_classification_fewshots_df.to_dict("records")
        ]
        return reject_classification_fewshots

    def load_checkers(self) -> list[Checker]:
        filename = self.load_domain_specific_config()['FILTER_CONFIG_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        filter_config_df = pd.read_csv(path_to_csv, encoding='latin1')
        checkers_list = []
        for row in filter_config_df.to_dict("records"):
            if row['type_of_filter'].strip() == "exact word matching":
                checkers_list.append(ExactWordMatchingChecker(
                    row['key_in_state'].split(","), row['metadata_field']))

            elif row['type_of_filter'].strip() == "item":
                checkers_list.append(ItemChecker(
                    row['key_in_state'], row['metadata_field']))

            elif row['type_of_filter'].strip() == "location":
                checkers_list.append(LocationChecker(
                    row['key_in_state'], row['metadata_field'].split(","), row['default_max_distance_in_km'],
                    row['distance_type'], GoogleV3Wrapper()))

            elif row['type_of_filter'].strip() == "value range":
                checkers_list.append(ValueRangeChecker(row['key_in_state'], row['metadata_field']))

            elif row['type_of_filter'].strip() == "word in":
                checkers_list.append(WordInChecker(
                    row['key_in_state'].split(","), row['metadata_field']))

        return checkers_list
