from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np
from src.car_price_prediction.utlis import save_pkl, evaluate_the_models, top_n_models_with_tuning

# Importing all models
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    model_path = Path("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def initiate_model_trainer(self, x_train, x_test, y_train, y_test, top_n=3):
        try:
            logging.info("Initiating the Model Trainer")
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "SVC": SVR(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "ExtraTreeRegressor": ExtraTreeRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=50, random_state=2),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=3),
                "AdaBoostRegressor": AdaBoostRegressor(n_estimators=50, random_state=3),
                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, random_state=3),
                "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
                "BaggingRegressor": BaggingRegressor(n_estimators=50, random_state=3),
                "KNeighborsRegressor": KNeighborsRegressor(),
            }
            logging.info("Starting the model evaluation")
            evaluated_models = evaluate_the_models(models, x_train, x_test, y_train, y_test)
            logging.info("Evaluation of models completed")
            logging.info(f"Every model performance=>\n" + "\n".join([str(i) for i in evaluated_models.items()]))

            # Find the top N models and perform hyperparameter tuning
            logging.info(f"Finding the top {top_n} models and tuning them")
            best_model_name, best_model_trained, best_scores = top_n_models_with_tuning(
                models, evaluated_models, x_train, y_train, x_test, y_test, n=top_n
            )
            logging.info(f"Tuned top {top_n} models successfully")

            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(f"Best Model Scores: {best_scores}")
            logging.info("Saving the best model")
            save_pkl(obj=best_model_trained, obj_path=self.config.model_path)
            logging.info("Model Pickle file is saved")
            

        except Exception as e:
            logging.error("Error occurred during model training: {}".format(e))
            raise CustomException(e, sys)