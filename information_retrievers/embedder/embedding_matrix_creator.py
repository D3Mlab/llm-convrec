import os

import pandas as pd
import faiss
import torch
from tqdm import tqdm
from information_retrievers.embedder.bert_embedder import BERT_model


class EmbeddingMatrixCreator:
    def __init__(self, model: BERT_model):
        self.embedding_model = model

    def create_embedding_matrix_from_reviews(self, reviews_df: pd.DataFrame, output_filepath=None, batch_size=128,
                                             k=10) -> torch.Tensor:
        reviews = reviews_df['Review']

        if output_filepath is None or not os.path.exists(output_filepath):
            start_index = 0
            prev_matrix = None
        else:
            prev_matrix = torch.load(output_filepath)
            start_index = prev_matrix.size()[0]

        embeddings = []
        save_number = k * batch_size
        for i in tqdm(range(start_index, len(reviews), batch_size)):
            embedding = self.embedding_model.embed(reviews[i:i + batch_size].to_list())
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
    def create_embedding_matrix_from_database(database: faiss.Index):
        num_embeddings = database.ntotal
        embeddings = []
        for i in range(num_embeddings):
            embedding = torch.tensor(database.reconstruct(i))
            embeddings.append(embedding)
        return torch.stack(embeddings)

