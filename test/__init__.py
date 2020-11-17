from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from pickle_model import pickle_model
from joblib_model import joblib_model
import time
import pandas as pd

def run_test():
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # fit the model
    forest = RandomForestClassifier()
    forest.fit(X,y)

    # Save model with pickle
    pickle_save_start_time = time.time()
    pickle_model(forest, save=True, filepath='pickle.pkl')
    pickle_save_end_time = time.time()

    pickle_save_time = pickle_save_end_time - pickle_save_start_time


    pickle_load_start_time = time.time()
    forest_pickle_load = pickle_model(filepath = 'pickle.pkl')
    pickle_load_end_time = time.time()

    pickle_load_time = pickle_load_end_time - pickle_load_start_time

    pickle_time = [pickle_save_time, pickle_load_time]

    # Save the model with joblib
    joblib_save_start_time = time.time()
    joblib_model(forest, save=True, filepath='joblib.pkl')
    joblib_save_end_time = time.time()

    joblib_save_time = joblib_save_end_time - joblib_save_start_time


    joblib_load_start_time = time.time()
    forest_joblib_load = joblib_model(filepath = 'joblib.pkl')
    joblib_load_end_time = time.time()

    joblib_load_time = joblib_load_end_time - joblib_load_start_time

    joblib_time = [joblib_save_time, joblib_load_time]

    dict_return={
    'joblib': joblib_time,
    'pickle': pickle_time
    }
    return pd.DataFrame(dict_return, index=['Save','Load'])
