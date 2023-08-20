import os

import torch
import pandas as pd
import faiss
from tqdm import tqdm
from information_retriever.embedder.bert_embedder import BERT_model


class EmbeddingMatrixCreator:

    """
    Class responsible for creating matrix storing all the embeddings.

    :param model: model used to embed reviews
    """

    _embedding_model: BERT_model

    def __init__(self, model: BERT_model):
        self._embedding_model = model

    def create_embedding_matrix_from_reviews(self, reviews_df: pd.DataFrame, output_filepath=None, batch_size=128,
                                             k=10) -> torch.Tensor:
        """
        Create matrix by embedding reviews in reviews_df and save it to given output_filepath.
        If output_filepath is None, don't save the matrix.

        :param reviews_df: data frame that contains "Review" column storing all the reviews to embed
        :param output_filepath: file path to save the matrix
        :param batch_size: size of each batch when embedding
        :param k: number of batches in between saving the database
        """
        reviews = reviews_df['Review']

        if output_filepath is None or not os.path.exists(output_filepath):
            start_index = 0
            prev_matrix = None
        else:
            prev_matrix = torch.load(output_filepath)
            start_index = prev_matrix.size()[0]

        embeddings = []

        if not start_index == len(reviews):
            save_number = k * batch_size
            for i in tqdm(range(start_index, len(reviews), batch_size)):
                embedding = self._embedding_model.embed(reviews[i:i + batch_size].to_list())
                embedding = torch.as_tensor(embedding)
                embeddings.append(embedding)
                if output_filepath is not None and (i - start_index) % save_number == 0:
                    if prev_matrix is not None:
                        prev_matrix = torch.cat((prev_matrix, torch.cat(embeddings)))
                    else:
                        prev_matrix = torch.cat(embeddings)
                    torch.save(prev_matrix, output_filepath)
                    embeddings = []

        if not embeddings:
            return prev_matrix

        new_matrix = torch.cat(embeddings)
        if prev_matrix is not None:
            new_matrix = torch.cat((prev_matrix, new_matrix))

        if output_filepath is not None:
            torch.save(new_matrix, output_filepath)
        return new_matrix

    @staticmethod
    def create_embedding_matrix_from_database(database: faiss.Index, output_filepath=None) -> torch.Tensor:
        """
        Create embedding matrix storing all the embeddings of the reviews from vector database.

        :param database: vector database storing all embeddings
        :param output_filepath: file path to save the embedding matrix

        :return: The embedding matrix
        """
        num_embeddings = database.ntotal
        embeddings = []
        for i in range(num_embeddings):
            embedding = torch.tensor(database.reconstruct(i))
            embeddings.append(embedding)

        matrix = torch.stack(embeddings)
        if output_filepath is not None:
            torch.save(matrix, output_filepath)
        return matrix

