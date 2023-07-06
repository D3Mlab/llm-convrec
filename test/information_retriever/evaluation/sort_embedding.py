import pandas as pd

class SortEmbedding():
    def sort_embedding(self, source_file, destination_file):
        # load your data
        df = pd.read_csv(source_file)

        # sort the dataframe by the column of interest
        df = df.sort_values(by='business_id')

        # save your data back to csv
        df.to_csv(destination_file, index=False)