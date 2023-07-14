import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def load_processed_data(PROCESSED_DATA_PATH):
    """This function loads the preprocessed data.

    Args:
        INPUT_DIR: Path of the directory there preprocessed data resides.
    Returns:
        X_train, y_train, X_test and y_test
    """

    logging.info("Reading Data From :" + str(PROCESSED_DATA_PATH))

    X_train_filepath = os.path.join(PROCESSED_DATA_PATH, "X_train.csv")
    X_test_filepath = os.path.join(PROCESSED_DATA_PATH, "X_test.csv")
    y_train_filepath = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
    y_test_filepath = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

    # Loading the data
    logging.info("Loading Data")

    X_train = pd.read_csv(X_train_filepath)
    y_train = pd.read_csv(y_train_filepath)
    X_test = pd.read_csv(X_test_filepath)
    y_test = pd.read_csv(y_test_filepath)

    return X_train, y_train, X_test, y_test


# Function to Save Models
def model_dump(model, filepath):
    """This function dumps the trained model.

    Args:
        model: trained models.

        filepath: path to save/dump the model.
    Returns:
        None
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def train_linear_regression(X_train, y_train, MODEL_DIR, MODEL_NAME):
    """This function trains the Linear Regression Model.

    Args:
        X_train: Pandas Dataframe or Numpy Array that have features to train the model.

        y_train: Pandas Dataframe or numpy array that have target values.

        MODEL_DIR: Absolute or relative path of directory where the model will be saved.

        MODEL_NAME: Name of model that will be save as pickle file.
    Returns:
        None
    """
    logging.info("Linear Regression Model Will be Saved to : " + str(MODEL_DIR))
    # Linear Regression Model
    logging.info("Training Linear Regression Model")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    logging.info("Saving Model")
    model_dump(lin_reg, os.path.join(MODEL_DIR, MODEL_NAME))


def train_decision_tree(X_train, y_train, MODEL_DIR, MODEL_NAME):
    """This function trains the Decision Tree Model.

    Args:
        X_train: Pandas Dataframe or Numpy Array that have features to train the model.

        y_train: Pandas Dataframe or numpy array that have target values.

        MODEL_DIR: Absolute or relative path of directory where the model will be saved.

        MODEL_NAME: Name of model that will be save as pickle file.
    Returns:
        None
    """
    logging.info("Decision Tree Model Will be Saved to : " + str(MODEL_DIR))
    # Decision Tree Model
    logging.info("Training Decision Tree Model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    logging.info("Saving Model")
    model_dump(tree_reg, os.path.join(MODEL_DIR, MODEL_NAME))


def train_random_forest_RandomSearch(
    X_train, y_train, param_distribs, n_iter, cv, MODEL_DIR, MODEL_NAME
):
    """This function trains the Random Forest Model.

    RandomSearchCV method is used to find the best hyperparameters while training.

    Args:
        X_train: Pandas Dataframe or Numpy Array that have features to train the model.

        y_train: Pandas Dataframe or numpy array that have target values.

        param_distribs: dictionary that contains the parameters for hypertuning.

        n_iter: number of iterations that RandomSearchCV will used to find best estimator.

        cv: number of cross-validation folds.

        MODEL_DIR: Absolute or relative path of directory where the model will be saved.

        MODEL_NAME: Name of model that will be save as pickle file.
    Returns:
        None
    """
    logging.info("Training Random Forest (With Random Search)")

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train.values.ravel())

    cvres = rnd_search.cv_results_
    logging.info("Random Forest Results (With Random Search)")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logging.info(np.sqrt(-mean_score), params)

    logging.info("Random Forest Best Params with Random Search : " + str(rnd_search.best_params_))
    rf_random_search = rnd_search.best_estimator_
    logging.info("Saving Model")
    model_dump(rf_random_search, os.path.join(MODEL_DIR, MODEL_NAME))


def train_random_forest_GridSearch(X_train, y_train, param_grid, cv, MODEL_DIR, MODEL_NAME):
    """This function trains the Random Forest Model.

    GridSearchCV method is used to find the best hyperparameters while training.

    Args:
        X_train: Pandas Dataframe or Numpy Array that have features to train the model.

        y_train: Pandas Dataframe or numpy array that have target values.

        param_grid: list of dictionaries that contains the parameters for hypertuning.

        cv: number of cross-validation folds.

        MODEL_DIR: Absolute or relative path of directory where the model will be saved.

        MODEL_NAME: Name of model that will be save as pickle file.
    Returns:
        None
    """
    logging.info("Training Random Forest (With Grid Search)")

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train.values.ravel())

    cvres = grid_search.cv_results_
    logging.info("Random Forest Results (With Grid Search)")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logging.info(np.sqrt(-mean_score), params)

    logging.info("Random Forest Best Params with Grid Search : " + str(grid_search.best_params_))
    rf_gridsearch = grid_search.best_estimator_

    logging.info("Saving Model")
    model_dump(rf_gridsearch, os.path.join(MODEL_DIR, MODEL_NAME))
