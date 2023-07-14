import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def model_load(filepath):
    """This function loads the trained model.

    Args:
        filepath: path to load the model.
    Returns:
        None
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def check_if_file_exists(f):
    """This function checks if a file exists.

    Args:
        f: filepath that is needed to be checked.
    Returns:
        None
    Raises:
        Exits if file does not exists.
    """

    if os.path.exists(f) is False:
        logging.info(str(f) + " does not exists. Exiting...")
        exit()


def separator():
    """This function puts a separator of 50 "=" symbols"""
    logging.info("=" * 50)


def load_processed_data(PROCESSED_DATA_PATH):
    """This function loads the preprocessed data.

    Args:
        INPUT_DIR: Path of the directory there preprocessed data resides.
    Returns:
        X_train, y_train, X_test and y_test
    Raises:
        Exits if Processed data is not present.
    """

    logging.info("Reading Data From :" + str(PROCESSED_DATA_PATH))

    X_train_filepath = os.path.join(PROCESSED_DATA_PATH, "X_train.csv")
    X_test_filepath = os.path.join(PROCESSED_DATA_PATH, "X_test.csv")
    y_train_filepath = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
    y_test_filepath = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

    check_if_file_exists(X_train_filepath)
    check_if_file_exists(X_test_filepath)
    check_if_file_exists(y_train_filepath)
    check_if_file_exists(y_test_filepath)

    # Loading the data
    logging.info("Loading Data")

    X_train = pd.read_csv(X_train_filepath)
    y_train = pd.read_csv(y_train_filepath)
    X_test = pd.read_csv(X_test_filepath)
    y_test = pd.read_csv(y_test_filepath)

    return X_train, y_train, X_test, y_test


# Loading Models
def load_model(MODEL_DIR, MODEL_NAME):
    """This function loads a trained model that was saved as pickle object.

    Args:
        MODEL_DIR: Directory where model object is present.

        MODEL_NAME: Name of the model pickle file.
    Returns:
        model : loaded model
    Raises:
        Exists the code if model does not exists.
    """
    logging.info("Loading Model " + str(MODEL_NAME))
    check_if_file_exists(os.path.join(MODEL_DIR, MODEL_NAME))
    model = model_load(os.path.join(MODEL_DIR, MODEL_NAME))
    return model


def get_rmse(y_true, y_pred):
    """This function computes the 'root mean square error(RMSE)'.

    Args:
        y_true: Series or numpy array(1D) of actual values.

        y_pred: Series or numpy array(1D) of predicted values.
    Returns:
        RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Scoring Linear Regression Model
def score_model(trained_model, MODEL_NAME, X_train, y_train, X_test, y_test):
    """This function scores the model using RMSE as performance matric.

    Args:
        trained_model: trained model object.

        MODEL_NAME: model name for the purpose of logs.

        X_train: Pandas Dataframe or Numpy Array that have features using which model was trained.

        y_train: Pandas Dataframe or numpy array that have train-target values.

        X_test: Pandas Dataframe or Numpy Array that have features using which model will predict.

        y_test: Pandas Dataframe or numpy array that have test-target values.
    Returns:
        RMSE on train and test data.
    """
    train_rmse = get_rmse(trained_model.predict(X_train), y_train)
    test_rmse = get_rmse(trained_model.predict(X_test), y_test)
    logging.info(str(MODEL_NAME) + " Train RMSE : " + str(train_rmse))
    logging.info(str(MODEL_NAME) + " Test RMSE : " + str(test_rmse))
    separator()
    return train_rmse, test_rmse


def get_feature_importance(trained_model, MODEL_NAME, train_columns_list):
    """This function saves the feature importance of a model.

    Args:
        trained_model: trained model object.

        MODEL_NAME: model name for the purpose of logs.

        train_columns_list: list of features/columns on which model was trained.
    Returns:
        None
    """
    try:
        feature_importances = trained_model.feature_importances_
        feature_importances = sorted(zip(feature_importances, train_columns_list), reverse=True)
        logging.info(str(MODEL_NAME) + " Feature Importance")
        logging.info(feature_importances)
    except Exception:
        logging.info("feature importance method does not exists for the model " + str(MODEL_NAME))
    separator()
