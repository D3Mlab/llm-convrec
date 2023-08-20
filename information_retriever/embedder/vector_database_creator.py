import os

import torch
import faiss
from tqdm import tqdm
import pandas as pd

from information_retriever.embedder.bert_embedder import BERT_model


class VectorDatabaseCreator:

    """
    Class responsible for creating vector database storing all the embeddings.

    :param model: model used to embed reviews
    """

    _embedding_model: BERT_model

    def __init__(self, model: BERT_model):
        self._embedding_model = model

    def create_vector_database_from_reviews(self, reviews_df: pd.DataFrame, output_filepath=None, batch_size=128,
                                            k=10) -> faiss.Index:
        """
        Create faiss database by embedding reviews in reviews_df and save it to given output_filepath.
        If output_filepath is None, don't save the database.

        :param reviews_df: data frame that contains "Review" column storing all the reviews to embed
        :param output_filepath: file path to save the database
        :param batch_size: size of each batch when embedding
        :param k: number of batches in between saving the database
        """
        reviews = reviews_df['Review']

        if output_filepath is not None and os.path.exists(output_filepath):
            index = faiss.read_index(output_filepath)
            start_index = index.ntotal
        else:
            dimension_size = 768
            index = faiss.IndexFlatIP(dimension_size)
            start_index = 0

        if start_index < len(reviews):
            save_number = k * batch_size
            for i in tqdm(range(start_index, len(reviews), batch_size)):
                embedding = self._embedding_model.embed(reviews[i:i + batch_size].to_list())
                index.add(embedding)
                if output_filepath is not None and (i - start_index) % save_number == 0:
                    faiss.write_index(index, output_filepath)

            if output_filepath is not None:
                faiss.write_index(index, output_filepath)

        return index

    @staticmethod
    def create_vector_database_from_matrix(embedding_matrix: torch.Tensor, output_filepath=None) -> faiss.Index:
        """
        Create vector database storing all the embeddings of the reviws from embedding matrix.

        :param embedding_matrix: matrix storing all embeddings
        """
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix.numpy())
        if output_filepath is not None:
            faiss.write_index(index, output_filepath)
        return index
