from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os
import sys
from src.car_price_prediction.utlis import get_data_from_database_into_dataframe
from src.car_price_prediction.constant import *
import pandas as pd

@dataclass
class DataIngestionConfig:
    raw_data_path = "artifacts/raw_data.csv"
    test_data_path = "artifacts/test_data.csv"
    train_data_path = "artifacts/train_data.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion is Initiated")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # df = pd.read_csv('notebooks/data/main.csv',)
            #getting data from database
            df = get_data_from_database_into_dataframe(URI,DB_NAME,COLLECTION)
            logging.info("Spilting the dataset into train and test")

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=436)

            df.to_csv(self.config.raw_data_path, header=True, index=False)
            train_data.to_csv(self.config.train_data_path, header=True, index=False)
            test_data.to_csv(self.config.test_data_path, header=True, index=False)

            logging.info("Successfully stored the data")
            logging.info("Data Ingestion is Completed")
            # print(df.shape)
            # print(train_data.shape)
            # print(test_data.shape)
            return (self.config.train_data_path, self.config.test_data_path)

        except Exception as e:
            logging.error(f"Error occurred in data ingestion due to {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion())