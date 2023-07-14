import torch
import pandas as pd
from information_retrievers.ir.search_engine_old import NeuralSearchEngine

class InformationRetrievalRankCSV():
    def __init__(self, information_retrieval:NeuralSearchEngine):
        self.information_retrieval = information_retrieval
    def rank_item_to_csv(self, df_embedding:pd.DataFrame, matrix:torch.Tensor, item:torch.Tensor, source_file_path:str, destination_file_path:str):
        df_query = pd.read_csv(source_file_path)
        size = len(df_query["num_of_1"])

        # Create a new DataFrame of the same size as the original
        new_df = pd.DataFrame(index=df_query.index, columns=df_query.columns)

        # This loop gets each query and does information retrieval on it.
        for i in range(size):
            list_of_ranking = self.information_retrieval.search_for_topk(df_query["Unnamed: 0"][i], item.numel(), 1, df_embedding, matrix, item)

            # Loop through each column in the original DataFrame
            for col in df_query.columns:
                # If the column name exists in the list of rankings, find its index; else, assign -1
                index = (list_of_ranking.index(col) if col in list_of_ranking else -1)+1
                # Assign this index to the corresponding row in the column of the new DataFrame
                new_df.loc[i, col] = index

        for i in range(size):
            new_df.iloc[i, 0] = df_query["Unnamed: 0"][i]

        new_df = new_df.drop('num_of_1', axis=1)
        
        # Save the new DataFrame as a CSV file
        new_df.to_csv(destination_file_path, index=False)