from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
from src.car_price_prediction.utlis import load_pkl
import pandas as pd
import sys


@dataclass
class PredictionPipelineConfig:
    model_pkl_path: Path = Path('artifacts/model.pkl')
    preprocessor_pkl_path: Path = Path('artifacts/preprocessor.pkl')


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()

    def run_pipeline(self, name: str, company: str, year: int, kms_driven: int, fuel_type: str):
        try:
            logging.info("Starting Prediction Pipeline")

            # Load the trained model and preprocessor
            model = load_pkl(self.config.model_pkl_path)
            preprocessor = load_pkl(self.config.preprocessor_pkl_path)
            logging.info("Successfully loaded necessary files")
            print(preprocessor.get_feature_names_out())
            # Create a DataFrame from the input
            input_data = pd.DataFrame({
                "name": [name],
                "company": [company],
                "year": [year],
                "kms_driven": [kms_driven],
                "fuel_type": [fuel_type]
            })
            logging.info(f"Input data converted to DataFrame: {input_data}")

            # Preprocess the input data
            transformed_data = preprocessor.transform(input_data)
            logging.info(f"Data after preprocessing: {transformed_data}")

            # Predict the car price
            prediction = model.predict(transformed_data)
            logging.info(f"Prediction made: {prediction}")

            # Return the predicted price
            return prediction[0]

        except Exception as e:
            logging.error(f"Error occurred in Prediction Pipeline due to {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage of the PredictionPipeline
    obj = PredictionPipeline()
    try:
        result = obj.run_pipeline(
            name="Honda City",
            company="Honda",
            year=2015,
            kms_driven=50000,
            fuel_type="Petrol"
        )
        print(f"Predicted car price: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
