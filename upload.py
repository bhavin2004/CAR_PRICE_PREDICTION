from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import json
from src.car_price_prediction.constant import *

# MongoDB connection URI


# Create a new client and connect to the server
client = MongoClient(URI, server_api=ServerApi('1'))

# Read the CSV file
df = pd.read_csv('https://github.com/rajtilakls2510/car_price_predictor/raw/refs/heads/master/quikr_car.csv', encoding='latin1')

# Convert the DataFrame to JSON
data_json = json.loads(df.to_json(orient='records'))

# creating connection with database
connection = client[DB_NAME][COLLECTION]

# Insert JSON data into the collection
connection.insert_many(data_json)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
