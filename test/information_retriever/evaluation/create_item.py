import pandas as pd
import torch

class CreateItemSeperation():
    def get_item_seperation(self, source_file_path:str, destination_file_path:str):
        # Load your CSV file into a pandas DataFrame
        df = pd.read_csv(source_file_path)

        # Group by the specified column and count the number of rows in each group
        value_counts = df.groupby('item_id').size()

        #Convert it into a list
        value_counts = value_counts.to_list()

        #Change it into a tensor
        tensor = torch.tensor(value_counts)

        torch.save(tensor, destination_file_path)