import pandas as pd

class MRP():
    def find_MRP_score(self, relative_file_path:str, ranking_file_path:str, destination_file_path:str):
        #Go through the relative file to find all the related item's 
        df_relative = pd.read_csv(relative_file_path)
        df_ranking = pd.read_csv(ranking_file_path)

        positions = []
        length = df_relative.shape[0]
        width = df_relative.shape[1]

        for i in range(length):
            small_list = []
            for j in range(1, width-1):
                if(df_relative.iloc[i, j]==1):
                    small_list.append(df_ranking.iloc[i, j])
            
            small_list.sort()
            positions.append(small_list)
        
        list_of_MRP = []
        for i in range(len(positions)):
            counter = 0
            for j in range(len(positions[i])):
                if(positions[i][j]<=len(positions[i])):
                    counter += 1
                else:
                    break
            
            list_of_MRP.append(counter/len(positions[i]))

        # Add the list as a new column
        df = pd.read_csv(destination_file_path)
        df["MRP"] = list_of_MRP
        df.to_csv(destination_file_path, index=False)