import faiss
from tqdm import tqdm
import torch
import pandas as pd

from information_retrievers.embedder.bert_embedder import BERT_model


class VectorDatabaseCreator:
    def __init__(self, model: BERT_model):
        self.embedding_model = model

    def create_vector_database_from_reviews(self, reviews_df: pd.DataFrame, batch_size=128) -> faiss.Index:
        reviews = reviews_df['Review']
        dimension_size = 768
        index = faiss.IndexFlatIP(dimension_size)

        for i in tqdm(range(0, len(reviews), batch_size)):
            embedding = self.embedding_model.embed(reviews[i:batch_size].to_list())
            index.add(embedding)

        return index

    @staticmethod
    def create_vector_database_from_matrix(embedding_matrix: torch.Tensor) -> faiss.Index:
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix.numpy())
        return index


