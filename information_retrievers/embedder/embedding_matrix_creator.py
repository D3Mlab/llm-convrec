
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from information_retrievers.embedder.bert_embedder import BERT_model


class EmbeddingMatrixCreator:
    def __init__(self, model: BERT_model):
        self.embedding_model = model

    def create_embedding_matrix_from_reviews(self, reviews_df: pd.DataFrame, batch_size=128) -> torch.Tensor:
        reviews = reviews_df['Review']
        embeddings = []
        for i in tqdm(range(0, len(reviews), batch_size)):
            embedding = self.embedding_model.embed(reviews[i:batch_size].to_list())
            embedding = torch.as_tensor(embedding)
            embeddings.append(embedding)
        return torch.cat(embeddings)

    @staticmethod
    def create_embedding_matrix_from_database(database: faiss.Index):
        num_embeddings = database.ntotal
        embeddings = []
        for i in range(num_embeddings):
            embedding = torch.tensor(database.reconstruct(i))
            embeddings.append(embedding)
        return torch.stack(embeddings)
