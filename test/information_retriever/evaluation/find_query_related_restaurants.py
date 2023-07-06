#This cell reads from the PMD file and create a csv file that contains which query is related to which restaurant
import pandas as pd

class FindQueryRelatedRestaurants():
    # Function to build true labels
    def build_true_labels(self, source_file_path, destination_file_path):
        df = pd.read_csv(source_file_path)
        queries = df["query"].unique()
        restaurants = df["Restaurant name"].unique()
        final_df = pd.DataFrame(index=queries, columns=restaurants)
        for index, row in df.iterrows():
            final_df.at[row["query"], row["Restaurant name"]] = row["If only Low or  High"]
        
        final_df.to_csv(destination_file_path)
    
    def find_R_value(self, source_file_path, destination_file_path):
        # read the file
        df = pd.read_csv(source_file_path)

        # calculate the count of 1's in each row and add this as a new column
        df['num_of_1'] = df.apply(lambda row: sum(row == 1), axis=1)

        # save the updated DataFrame back to CSV
        df.to_csv(destination_file_path, index=False)