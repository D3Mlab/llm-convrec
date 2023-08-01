import re

import torch
import numpy as np
import faiss
import pandas as pd
import yaml

from information_retriever.embedder.bert_embedder import BERT_model
from information_retriever.embedder.embedding_matrix_creator import EmbeddingMatrixCreator
from information_retriever.embedder.vector_database_creator import VectorDatabaseCreator
from information_retriever.filter.filter import Filter
from information_retriever.filter.exact_word_matching_filter import ExactWordMatchingFilter
from information_retriever.filter.item_filter import ItemFilter
from information_retriever.filter.value_range_filter import ValueRangeFilter
from information_retriever.filter.word_in_filter import WordInFilter
from information_retriever.vector_database import VectorDataBase
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class DomainSpecificConfigLoader:

    """
    Class responsible for loading domain specific data.
    """

    def __init__(self, system_config: dict):
        self.system_config = system_config

    def load_domain(self) -> str:
        """
        Load domain name (e.g. restaurants)
        """
        return self._load_domain_specific_config()['DOMAIN']

    def load_constraints_categories(self) -> list[dict]:
        """
        Load constraints categories that defines the constraint details

        :return: constraints categories that defines the constraint details
        """
        constraints_category_filename = self._load_domain_specific_config()['CONSTRAINTS_CATEGORIES']
        path_to_csv = f'{self._get_path_to_domain()}/{constraints_category_filename}'
        constraints_df = pd.read_csv(path_to_csv, encoding='latin1', keep_default_na=False)
        return constraints_df.to_dict("records")

    def load_accepted_items_fewshots(self) -> list[dict]:
        """
        Load few shot examples that is used in accepted items extractor

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['ACCEPTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
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
        """
        Load few shot examples that is used in rejected items extractor

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['REJECTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
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
        """
        Load few shot examples that is used in current items extractor

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['CURRENT_ITEMS_EXTRACTOR_FEWSHOTS_FILE']
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
        """
        Load few shot examples that is used in constraints updater

        :return: list of few shot examples
        """
        constraints_updater_fewshot_filename = self._load_domain_specific_config()[
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
        """
        Load few shot examples that is used in extract category prompt in answer rec action

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['ANSWER_EXTRACT_CATEGORY_FEWSHOTS_FILE']
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
        """
        Load few shot examples that is used in prompt for answering question based on information retrieval

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['ANSWER_IR_FEWSHOTS_FILE']
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
        """
        Load few shot examples that is used in prompt for dividing up user input containing multiple questions
        to individual question

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['ANSWER_SEPARATE_QUESTIONS_FEWSHOTS_FILE']
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

    def _load_domain_specific_config(self) -> dict:
        """
        Load domain_specific_config.yaml.

        :return: dict representation of domain_specific_config.yaml.
        """
        path_to_domain = self._get_path_to_domain()

        with open(f'{path_to_domain}/domain_specific_config.yaml') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _get_path_to_domain(self) -> str:
        """
        Load path to folder containing domain specific configs.

        :return: path to folder containing domain specific configs.
        """
        return self.system_config['PATH_TO_DOMAIN_CONFIGS']

    def load_inquire_classification_fewshots(self) -> list[dict]:
        """
        Load few shot example used for user intent classification corresponding to inquire user intent

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['INQUIRE_CLASSIFICATION_FEWSHOTS_FILE']
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
        """
        Load few shot example used for user intent classification corresponding to accept recommendation user intent

        :return: list of few shot examples
        """
        filename = self._load_domain_specific_config()['ACCEPT_CLASSIFICATION_FEWSHOTS_FILE']
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
        """
        Load few shot example used for user intent classification corresponding to reject recommendation user intent
        """
        filename = self._load_domain_specific_config()['REJECT_CLASSIFICATION_FEWSHOTS_FILE']
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

    def load_filters(self) -> list[Filter]:
        """
        Load config details about metadata filtering used in information retrieval

        :return: list of filters to apply
        """
        filename = self._load_domain_specific_config()['FILTER_CONFIG_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        filter_config_df = pd.read_csv(path_to_csv, encoding='latin1')
        filters_list = []

        for row in filter_config_df.to_dict("records"):
            if row['type_of_filter'].strip() == "exact word matching":
                filters_list.append(ExactWordMatchingFilter(
                    [key.strip() for key in row['key_in_state'].split(",")], row['metadata_field'].strip()))

            elif row['type_of_filter'].strip() == "item":
                filters_list.append(ItemFilter(
                    row['key_in_state'].strip(), row['metadata_field'].strip()))

            elif row['type_of_filter'].strip() == "value range":
                filters_list.append(ValueRangeFilter(row['key_in_state'].strip(), row['metadata_field'].strip()))

            elif row['type_of_filter'].strip() == "word in":
                filters_list.append(WordInFilter(
                    [key.strip() for key in row['key_in_state'].split(",")], row['metadata_field'].strip()))

        return filters_list

    def load_item_metadata(self) -> pd.DataFrame:
        """
        Load metadata of all items

        :return: metadata of all items
        """
        filename = self._load_domain_specific_config()['PATH_TO_ITEM_METADATA']
        path_to_items_metadata = f'{self._get_path_to_domain()}/{filename}'
        return pd.read_json(path_to_items_metadata, orient='records', lines=True)

    def load_data_for_pd_search_engine(self) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Load data (item id corresponding to each review, review texts and embedding matrix)
        used for initializing PD Search Engine

        :return: data used for initializing pd search engine
        """
        path_to_domain = self._get_path_to_domain()
        filename = self._load_domain_specific_config()['PATH_TO_REVIEWS']
        filepath = f'{self._get_path_to_domain()}/{filename}'
        reviews_df = pd.read_csv(filepath)

        # load embedding matrix
        embedding_matrix_filename = self._load_domain_specific_config()['PATH_TO_EMBEDDING_MATRIX']
        path_to_embedding_matrix = f'{path_to_domain}/{embedding_matrix_filename}'
        embedding_matrix = self._create_embedding_matrix(reviews_df, path_to_embedding_matrix)

        review_item_ids = reviews_df["item_id"].to_numpy()
        reviews = reviews_df["Review"].to_numpy()
        return review_item_ids, reviews, embedding_matrix

    def load_data_for_vector_database_search_engine(self) -> tuple[np.ndarray, np.ndarray, VectorDataBase]:
        """
        Load data (item id corresponding to each review, review texts and faiss database)
        for initializing vector database search engine

        :return: data used for initializing vector database search engine
        """
        filename = self._load_domain_specific_config()['PATH_TO_REVIEWS']
        filepath = f'{self._get_path_to_domain()}/{filename}'
        reviews_df = pd.read_csv(filepath)

        path_to_domain = self._get_path_to_domain()
        database_filename = self._load_domain_specific_config()['PATH_TO_DATABASE']
        path_to_database = f'{path_to_domain}/{database_filename}'

        database = self._create_database(reviews_df, path_to_database)

        review_item_ids = reviews_df["item_id"].to_numpy()
        reviews = reviews_df["Review"].to_numpy()
        return review_item_ids, reviews, VectorDataBase(database)

    def _create_database(self, reviews_df: pd.DataFrame, path_to_database: str) -> faiss.Index:
        """
        Create or load vector database. If database already exists in path_to_database, load database. Otherwise,
        create database and save them to path_to_database from embedding matrix or reviews_df.

        :param reviews_df: dataframe containing reviews
        :param path_to_database: path to vector database
        :return: FAISS database
        """
        # initialize vector database creator
        model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        bert_model = BERT_model(model_name, model_name)
        vector_database_creator = VectorDatabaseCreator(bert_model)

        # load file path to embedding matrix
        path_to_domain = self._get_path_to_domain()
        domain_specific_config = self._load_domain_specific_config()
        reviews_embedding_matrix_filename = domain_specific_config['PATH_TO_EMBEDDING_MATRIX']
        path_to_embedding_matrix = f'{path_to_domain}/{reviews_embedding_matrix_filename}'

        # create database
        if not os.path.exists(path_to_database) and os.path.exists(path_to_embedding_matrix):
            embedding_matrix = torch.load(path_to_embedding_matrix)
            if embedding_matrix.shape[0] == reviews_df.shape[0]:
                database = vector_database_creator.create_vector_database_from_matrix(embedding_matrix, path_to_database)
            else:
                database = vector_database_creator.create_vector_database_from_reviews(reviews_df, path_to_database)
        else:
            database = vector_database_creator.create_vector_database_from_reviews(reviews_df, path_to_database)
        return database

    def _create_embedding_matrix(self, reviews_df: pd.DataFrame, path_to_embedding_matrix: str) -> torch.Tensor:
        """
        Create or load matrix containing embedding matrix. If embedding matrix already exists in
        path_to_embedding_matrix, load embedding matrix.
         Otherwise, create embedding matrix and save them to path_to_database from vector database or reviews_df.

        :param reviews_df: dataframe containing reviews
        :param path_to_embedding_matrix: path to embedding matrix
        :return: embedding matrix
        """
        # initialize embedding matrix creator
        model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        bert_model = BERT_model(model_name, model_name)
        embedding_matrix_creator = EmbeddingMatrixCreator(bert_model)

        # load file path to database
        path_to_domain = self._get_path_to_domain()
        domain_specific_config = self._load_domain_specific_config()
        database_filename = domain_specific_config['PATH_TO_DATABASE']
        path_to_database = f'{path_to_domain}/{database_filename}'

        # create embedding matrix
        if not os.path.exists(path_to_embedding_matrix) and os.path.exists(path_to_database):
            database = faiss.read_index(path_to_database)
            if database.ntotal == reviews_df.shape[0]:
                embedding_matrix = embedding_matrix_creator.create_embedding_matrix_from_database(
                    database,
                    path_to_embedding_matrix
                )
            else:
                embedding_matrix = embedding_matrix_creator.create_embedding_matrix_from_reviews(
                    reviews_df,
                    path_to_embedding_matrix
                )
        else:
            embedding_matrix = embedding_matrix_creator.create_embedding_matrix_from_reviews(
                reviews_df,
                path_to_embedding_matrix
            )
        return embedding_matrix

    def load_hard_coded_responses(self) -> list[dict]:
        """
        Load config that defines hard coded response.

        :return: config that defines hard coded response.
        """
        filename = self._load_domain_specific_config()['HARD_CODED_RESPONSES_FILE']
        path_to_csv = f'{self._get_path_to_domain()}/{filename}'
        responses_df = pd.read_csv(path_to_csv, encoding='latin1')
        responses = [
            {
                'action': row['action'],
                'response': row['response'],
                'constraints': row['constraints'].split(', ') if isinstance(row['constraints'], str) else []
            }
            for row in responses_df.to_dict("records")
        ]
        return responses

    def load_explanation_metadata_blacklist(self) -> list[str]:
        """
        load and return list of metadata keys that should be IGNORED when explanation recommended item

        :return: list of metadata keys that should be IGNORED when explanation recommended item
        """
        return self._load_domain_specific_config()['EXPLANATION_METADATA_BLACKLIST']

    @staticmethod
    def _load_dict_in_cell(data_string: str) -> dict:
        """
        load dict in csv cell format as following:

        key1=[value1, value2], key2=[value3]

        :param data_string: string in the csv cell
        :return: dict loaded from the given string
        """
        pattern = r'([^,]+)\s*=\s*\[([^\]]+)\]'
        matches = re.findall(pattern, data_string)
        data_dict = {key.strip(): [value.strip().removesuffix('"').removeprefix('"').removesuffix('.').strip().lower() for value in
                     re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lst)] for key, lst in matches}
        return data_dict
