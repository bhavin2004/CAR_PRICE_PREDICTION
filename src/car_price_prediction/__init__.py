from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
import pickle
import os, sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score
from time import time
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def get_data_from_database_into_dataframe(URI,DB_NAME,COLLECTION_NAME)->pd.DataFrame():
    try:
        # Create a new client and connect to the server
        client = MongoClient(URI, server_api=ServerApi('1'))
        connection = client[DB_NAME][COLLECTION_NAME]
        data = connection.find()
        df= pd.DataFrame(data)
        df.drop(columns='_id',axis=1,inplace=True)
        return df
    except Exception as e:
        raise CustomException(e,sys)


def save_pkl(obj, obj_path):
    try:
        folder_dir = os.path.dirname(obj_path)
        os.makedirs(folder_dir, exist_ok=True)
        logging.info(f"Initiating the Saving of Pickle file {obj_path}")
        with open(obj_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.error(
            "Error occurred in saving the {} pkl file: {}".format(obj_path, str(e))
        )
        raise CustomException(e, sys)


def evaluate_the_models(models: dict, x_train, x_test, y_train, y_test):
    try:
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        model_score = dict()
        with tqdm(total=len(models), desc="Evaluating Models", unit="model") as pbar:
            for model_name, model in models.items():
                now = time()
                model.fit(x_train, y_train)
                logging.info(
                    "Time Taken by model {} to fit the data: {:.2f} seconds".format(
                        model_name, time() - now
                    )
                )
                y_pred = model.predict(x_test)
                score = calculate_model_score(y_test, y_pred)
                model_score[model_name] = score
                pbar.update(1)  # Update progress bar after each model
        return model_score
    except Exception as e:
        logging.error("Error occurred during evaluation of all models: {}".format(e))
        raise CustomException(e, sys)


def calculate_model_score(true, predicted):
    # Calculate the accuracy score
    accuracy = accuracy_score(true, predicted)
    # Calculate the precision
    precision = precision_score(true, predicted)
    return {"accuracy": accuracy, "precision": precision}


def best_model(models: dict, evaluated_models: dict):
    try:
        score_type = "precision"
        max_score = 0
        best_model_name = None
        best_model = None
        for i in range(len(list(models.values()))):
            model_scores = list(evaluated_models.values())[i]
            score = model_scores[score_type]
            if score > max_score:
                best_model_name = list(evaluated_models.keys())[i]
                max_score = score
                best_model = list(models.values())[i]
        return best_model_name, best_model, max_score
    except Exception as e:
        logging.error(
            "Error occurred during calculating the best model score: {}".format(e)
        )
        raise CustomException(e, sys)


def load_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logging.error("Error occurred during loading the pkl file: {}".format(e))
        raise CustomException(e, sys)

def get_hyperparameter_grids():
    """
    Returns optimized hyperparameter grids for all models.
    """
    return {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["liblinear", "saga"],
        },
        "SVC": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"],
            "degree": [3, 5],
        },
        "MultinomialNB": {
            "alpha": [0.01, 0.1, 0.5, 1.0],
        },
        "BernoulliNB": {
            "alpha": [0.01, 0.1, 0.5, 1.0],
            "binarize": [0.0, 0.1, 0.2],
        },
        "GaussianNB": {},
        "DecisionTreeClassifier": {
            "max_depth": [None, 5, 10, 20, 50],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        },
        "ExtraTreeClassifier": {
            "max_depth": [None, 5, 10, 20, 50],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
        "ExtraTreesClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
        "AdaBoostClassifier": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
        },
        "GradientBoostingClassifier": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
        },
        "HistGradientBoostingClassifier": {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "min_samples_leaf": [20, 50, 100],
        },
        "BaggingClassifier": {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.5, 0.8, 1.0],
            "max_features": [0.5, 0.8, 1.0],
            "bootstrap": [True, False],
        },
        "KNeighborsClassifier": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
    }

def top_n_models_with_tuning(models, evaluated_models, x_train, y_train, x_test, y_test, n=3):
    """
    Tune the top N models based on evaluation and return the best model.
    """
    # Sort models based on precision and accuracy scores
    sorted_models = sorted(
        evaluated_models.items(),
        key=lambda x: (x[1]["precision"], x[1]["accuracy"]),
        reverse=True,
    )

    top_n_models = sorted_models[:n]
    best_model = None
    best_score = -1
    best_model_name = None

    # Hyperparameter grids for tuning
    param_grids = get_hyperparameter_grids()

    # Iterate over top N models and tune them
    for model_name, _ in top_n_models:
        model = models[model_name]
        param_grid = param_grids.get(model_name)

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Get the best model from grid search
        tuned_model = grid_search.best_estimator_
        tuned_model_score = grid_search.best_score_

        # Evaluate the tuned model on the test set
        y_pred = tuned_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        logging.info(f"Model: {model_name}, Tuned Accuracy: {accuracy:.2f}, Tuned Precision: {precision:.2f}")

        # Update best model if this one is better
        if precision > best_score:
            best_score = precision
            best_model = tuned_model
            best_model_name = model_name

    return best_model_name, best_model, best_score


def transform_text(text:str,ps=PorterStemmer()):
        text = text.lower()
        text = nltk.word_tokenize(text)
        res=[]
        for i in text:
            if i.isalpha():
                res.append(i)
        text = res[:]
        res.clear()
        for i in text:
            if i not in stopwords.words("english") and i not in punctuation:
                res.append(ps.stem(i))
        return " ".join(res)
