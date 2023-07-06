#Create the matrix for our information retrieval
import pandas as pd
import torch
import ast

class CreateMatrix():
    def create_matrix(self, source_file, destination_file):
        # Loop through the sorted embedding csv file
        df=pd.read_csv(source_file)

        container=[]

        size=len(df["review_text"])

        for i in range(size):
            embedding=ast.literal_eval(df["review_text"][i])
            embedding=torch.tensor(embedding)
            container.append(embedding)

        container=torch.stack(container)

        torch.save(container, destination_file)