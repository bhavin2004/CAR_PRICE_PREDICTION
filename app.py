import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.car_price_prediction.pipelines.predicrion_pipeline import PredictionPipeline
from src.car_price_prediction.pipelines.training_pipeline import Training_Pipeline

# Load cleaned car data
car = pd.read_csv('artifacts/cleaned_data.csv')

# Initialize Flask app
app = Flask(__name__)

# ------------------------------
# ✅ Form route
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')
# ------------------------------
# ✅ Form route
# ------------------------------
@app.route('/form')
def form():
    """Render home page"""
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('form.html', companies=companies, car_models=[], years=years, fuel_types=fuel_types)

# ------------------------------
# ✅ Train route
# ------------------------------
@app.route('/train')
def train():
    """Train the model"""
    try:
        obj_train = Training_Pipeline()
        obj_train.run_pipeline()
        return '<h1>The model has been trained successfully!</h1>'
    except Exception as e:
        return f'<h1>Error occurred during training: {e}</h1>'

# ------------------------------
# ✅ Get models based on selected company (AJAX)
# ------------------------------
@app.route('/get_models', methods=['POST'])
def get_models():
    """API to get car models based on selected company"""
    data = request.json
    selected_company = data.get('company')
    if selected_company and selected_company != 'Select Company':
        car_models = sorted(car[car['company'] == selected_company]['name'].unique().tolist())
    else:
        car_models = []

    return jsonify({'models': car_models})

# ------------------------------
# ✅ Predict route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict car price"""
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

        # Run prediction pipeline
        prediction_pipeline = PredictionPipeline()
        predicted_price = prediction_pipeline.run_pipeline(
            name=name,
            company=company,
            year=year,
            kms_driven=kms_driven,
            fuel_type=fuel_type
        )

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

# ------------------------------
# ✅ Run Flask App
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
