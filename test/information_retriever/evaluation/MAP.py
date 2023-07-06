import pandas as pd

class MAP():
    def find_MAP_score(self, relative_file_path:str, ranking_file_path:str, destination_file_path:str):
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
        
        list_of_MAP = []
        
        for i in range(len(positions)):
            counter = 1
            sum = 0
            for j in range(len(positions[i])):
                sum += counter/positions[i][j]
                counter += 1
            sum = sum/len(positions[i])
            
            list_of_MAP.append(sum)

        # Add the list as a new column
        list_of_MAP = pd.Series(list_of_MAP)
        list_of_MAP.name = "MAP"
        list_of_MAP = list_of_MAP.to_frame()
        list_of_MAP.to_csv(destination_file_path, index=False)