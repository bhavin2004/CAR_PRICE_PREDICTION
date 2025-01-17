from flask import Flask, render_template, request, redirect
from src.car_price_prediction.pipelines.predicrion_pipeline import PredictionPipeline
from src.car_price_prediction.pipelines.training_pipeline import Training_Pipeline
import pandas as pd

# Load cleaned car data
car = pd.read_csv('artifacts/cleaned_data.csv')

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Train route
@app.route('/train')
def train():
    try:
        obj_train = Training_Pipeline()
        obj_train.run_pipeline()
        return '<h1>The model has been trained successfully!</h1>'
    except Exception as e:
        return f'<h1>Error occurred during training: {e}</h1>'

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            company = request.form['company']
            year = int(request.form['year'])
            kms_driven = int(request.form['kms_driven'])
            fuel_type = request.form['fuel_type']

            # Prepare data for prediction
            input_data = pd.DataFrame({
                'name': [name],
                'company': [company],
                'year': [year],
                'kms_driven': [kms_driven],
                'fuel_type': [fuel_type]
            })

            # Prediction pipeline
            prediction_pipeline = PredictionPipeline()
            predicted_price = prediction_pipeline.run_pipeline(name=name,company=company,year=year,kms_driven=kms_driven,fuel_type=fuel_type)

            return render_template(
                'result.html',
                predicted_price=f"{predicted_price:.2f}",
                name=name,
                company=company,
                year=year,
                kms_driven=kms_driven,
                fuel_type=fuel_type
            )
        except Exception as e:
            return f'<h1>Error occurred during prediction: {e}</h1>'

    return render_template(
        'form.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

if __name__ == "__main__":
    app.run()
