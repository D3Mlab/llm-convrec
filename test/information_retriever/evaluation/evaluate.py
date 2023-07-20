import os
import pandas as pd
import scipy.stats
import torch
import numpy as np
from information_retrievers.embedder.bert_embedder import BERT_model
from find_query_related_restaurants import FindQueryRelatedRestaurants
from embedding_creator import EmbeddingCreator
from sort_embedding import SortEmbedding
from create_matrix import CreateMatrix
from create_item import CreateItemSeperation
from information_retrievers.ir.search_engine_old import NeuralSearchEngine
from information_retriever_ranking import InformationRetrievalRankCSV
from MAP import MAP
from MRP import MRP

def check_file_exists(file_path):
    return os.path.isfile(file_path)

class Evaluate():
    def __init__(self, model_name:str):
        self.embedding_model = BERT_model(model_name, model_name)
        self.query_related_restaurants = FindQueryRelatedRestaurants()
        self.embed_review = EmbeddingCreator(self.embedding_model)
        self.sort_embedding = SortEmbedding()
        self.matrix_creator = CreateMatrix()
        self.item_creator = CreateItemSeperation()
        self.information_retrieval = NeuralSearchEngine(self.embedding_model)
        self.information_retrieval_ranking = InformationRetrievalRankCSV(self.information_retrieval)
        self.get_map = MAP()
        self.get_mrp = MRP()
    
    def mean_confidence_interval(self, data, confidence=0.90):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return h

    def output(self, source_file_path:str):
        df = pd.read_csv(source_file_path)

        MAP_mean = df["MAP"].mean()
        MRP_mean = df["MRP"].mean()

        print("The MAP is: ", MAP_mean)
        print("The MRP is: ", MRP_mean)
        print("Confidence interval of MAP is: ", MAP_mean-self.mean_confidence_interval(df["MAP"].tolist()), "to", MAP_mean+self.mean_confidence_interval(df["MAP"].tolist()))
        print("Confidence interval of MRP is: ", MRP_mean-self.mean_confidence_interval(df["MRP"].tolist()), "to", MRP_mean+self.mean_confidence_interval(df["MRP"].tolist()))

    def evaluate(self, file_path_PMD, file_path_review):
        if(check_file_exists(file_path_PMD)):
            #Only create the file if it doesn't already exist
            if(not check_file_exists("data/PMD_Relativity.csv")):
                self.query_related_restaurants.build_true_labels(file_path_PMD, "data/PMD_Relativity.csv")
        else:
            #If there is no PMD file, exit immediately
            print("The PMD file does not exist")
            return None

        if(not check_file_exists("data/PMD_Relativity_With_R.csv")):
            self.query_related_restaurants.find_R_value("PMD_Relativity.csv", "data/PMD_Relativity_With_R.csv")
        
        #Get the embedding for all the reviews
        if(check_file_exists(file_path_review)):
            if(not check_file_exists("data/embedded_file.csv")):
                self.embed_review.embed(file_path_review, "data/embedded_file.csv")
        else:
            #If there is no PMD file, exit immediately
            print("The 50_restaurants_all_rates.csv file does not exist")
            return None
        
        if(not check_file_exists("data/embedded_file_sorted.csv")):
            self.sort_embedding.sort_embedding("data/embedded_file.csv", "data/embedded_file_sorted.csv")

        #Create the matrix and item pt file
        if(not check_file_exists("data/matrix.pt")):
            self.matrix_creator.create_matrix("data/embedded_file_sorted.csv", "data/matrix.pt")
        
        if(not check_file_exists("data/item.pt")):
            self.item_creator.get_item_seperation("data/embedded_file_sorted.csv", "data/item.pt")
        
        #Create the matrix and item tensor
        matrix = torch.load("data/matrix.pt")
        item = torch.load("data/item.pt")

        df = pd.read_csv("data/embedded_file_sorted.csv")

        if(not check_file_exists("data/information_retrieval_ranking.csv")):
            self.information_retrieval_ranking.rank_item_to_csv(df, matrix, item, "data/PMD_Relativity_With_R.csv", "data/information_retrieval_ranking.csv")

        if(not check_file_exists("data/evaluation.csv")):
            self.get_map.find_MAP_score("data/PMD_Relativity_With_R.csv", "data/information_retrieval_ranking.csv", "data/evaluation.csv")
            self.get_mrp.find_MRP_score("data/PMD_Relativity_With_R.csv", "data/information_retrieval_ranking.csv", "data/evaluation.csv")

        self.output("evaluation.csv")
        
if __name__ == "__main__":
    my_obj = Evaluate("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    my_obj.evaluate("/data/PMD.csv", "/data/50_restaurants_all_rates.csv")