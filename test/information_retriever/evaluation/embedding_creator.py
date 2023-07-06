import pandas as pd
import torch
from information_retrievers.neural_ir.neural_embedder import BERT_model

class EmbeddingCreator():
    def __init__(self, embedding_model:BERT_model):
        self.embedder = embedding_model 
    
    def embed(self, source_file_path:str, destination_file_path:str):
        # load the csv file
        df = pd.read_csv(source_file_path)

        # loop through the DataFrame, accessing one row at a time
        for index, row in df.iterrows():
            print(index)
            # let's assume you want to modify column 'ColumnName'
            value = row['review_text']

            # apply some function to modify the value
            embedding=self.embedder.embed([value])
            embedding=torch.tensor(embedding)
            embedding=embedding.squeeze(0)
            embedding=embedding.tolist()

            # update the value in the DataFrame
            df.loc[index, 'review_text'] = str(embedding)
        
        # save the DataFrame back to csv
        df.to_csv(destination_file_path, index=False)