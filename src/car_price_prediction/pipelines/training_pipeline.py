import sys
from src.logger import logging
from src.exception import CustomException
from src.car_price_prediction.components.data_ingestion import DataIngestion
from src.car_price_prediction.components.data_transformation import DataTransformation
from src.car_price_prediction.components.model_trainer import ModelTrainer



class Training_Pipeline():
    def __init__(self):
        self.data_ingestion=DataIngestion()
        self.data_transformation=DataTransformation()
        self.model_trainer=ModelTrainer()
    
    def run_pipeline(self):
        try:
            
            logging.info('Running Training Pipeline')
            train_path,test_path =self.data_ingestion.initiate_data_ingestion()
            print(train_path,test_path)
            x_train,x_test,y_train,y_test=self.data_transformation.initiate_data_transformation(train_path,test_path)
            # print(x_train,x_test,y_train,y_test)
            self.model_trainer.initiate_model_trainer(x_train,x_test,y_train,y_test)
            
        except Exception as e:
            
            logging.error(f"Error occurred in training pipeline due to {e}")
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = Training_Pipeline()
    obj.run_pipeline()