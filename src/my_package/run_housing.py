import argparse
import json
import logging
import os

import ingest
import mlflow
import mlflow.sklearn

# import numpy as np
# import pandas as pd
import score
import train
from prettytable import PrettyTable
from scipy.stats import randint

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_dir", help="enter the name of directory to save the data")
parser.add_argument("--processed_data_dir", help="Please enter input directory")
parser.add_argument("--model_dir", help="Please enter directory to save model pickles")


args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

# Setting Logger
LOG_DIR = config["log_dir"]
LOGPATH = os.path.join(LOG_DIR, "runall.log")
logging.basicConfig(filename=LOGPATH, format="%(asctime)s %(message)s", filemode="w")
logger = logging.getLogger()
# threshold of logger
logger.setLevel(logging.DEBUG)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# Load HOUSING_PATH from config file if not provided as user argument
if args.raw_data_dir is None:
    HOUSING_PATH = config["raw_data_dir"]
else:
    HOUSING_PATH = args.raw_data_dir

# Load PROCESSED_DATA_PATH from config file if not provided as user argument
if args.processed_data_dir is None:
    PROCESSED_DATA_PATH = config["processed_data_dir"]
else:
    PROCESSED_DATA_PATH = args.processed_data_dir

# Load MODEL_DIR from config file if not provided as user argument
if args.model_dir is None:
    MODEL_DIR = config["model_dir"]
else:
    MODEL_DIR = args.model_dir

# URL from where data will be downloaded
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Create Directory if does not exists to save downloaded data.
if os.path.exists(HOUSING_PATH) is False:
    os.makedirs(HOUSING_PATH)
# Create Directory if does not exists to save processed data.
if os.path.exists(PROCESSED_DATA_PATH) is False:
    os.makedirs(PROCESSED_DATA_PATH)

XTRAIN_FILEPATH = os.path.join(PROCESSED_DATA_PATH, "X_train.csv")
XTEST_FILEPATH = os.path.join(PROCESSED_DATA_PATH, "X_test.csv")
YTRAIN_FILEPATH = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
YTEST_FILEPATH = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

remote_server_uri = "http://localhost:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

# Put a name for your experiment
exp_name = "Run_Housing"
# mlflow.set_experiment(exp_name)
try:
    # Create the experiment if not present
    experiment_id = mlflow.create_experiment(exp_name)
except Exception:
    # If experiment is already present then get the experiment_id
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

with mlflow.start_run(run_name="Parent_HousingRun", experiment_id=experiment_id):
    with mlflow.start_run(
        run_name="Child_HousingRun_Ingest", experiment_id=experiment_id, nested=True
    ):
        ingest.fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
        logging.info("Data Download was successfull.")

        logging.info("Loading Data")

        housing = ingest.load_housing_data(HOUSING_PATH)
        test_size = 0.2
        train_df, test_df = ingest.train_test_split(housing, test_size=test_size, random_state=42)
        ingest.feature_engineering(
            train_set=train_df,
            test_set=test_df,
            XTRAIN_FILEPATH=XTRAIN_FILEPATH,
            YTRAIN_FILEPATH=YTRAIN_FILEPATH,
            XTEST_FILEPATH=XTEST_FILEPATH,
            YTEST_FILEPATH=YTEST_FILEPATH,
        )
        mlflow.log_param(key="housing_url", value=HOUSING_URL)
        mlflow.log_param(key="housing_path", value=HOUSING_PATH)
        mlflow.log_param(key="test_size", value=test_size)
        mlflow.log_artifact(os.path.join(HOUSING_PATH, "housing.csv"))

    if os.path.exists(PROCESSED_DATA_PATH) is False:
        logging.error(
            "Processed Data Directory Does Not Exists. \
                Please Check if data ingestion happend correctly. Exiting...."
        )
        exit()

    # Create Directory if does not exists to save trained models.
    if os.path.exists(MODEL_DIR) is False:
        os.makedirs(MODEL_DIR)

    with mlflow.start_run(
        run_name="Child_HousingRun_Train", experiment_id=experiment_id, nested=True
    ):
        # Load Processed Data
        X_train, y_train, X_test, y_test = train.load_processed_data(PROCESSED_DATA_PATH)

        # train LR
        lr = "linear_regression.pkl"
        train.train_linear_regression(X_train, y_train, MODEL_DIR, lr)

        # train DT
        dt = "decision_tree.pkl"
        train.train_decision_tree(X_train, y_train, MODEL_DIR, dt)

        # train RF with RandomSearch
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }
        rf_rs = "random_forest_random_search.pkl"
        n_iter = 10
        cv = 5
        train.train_random_forest_RandomSearch(
            X_train, y_train, param_distribs, n_iter, cv, MODEL_DIR, rf_rs
        )

        # train RF with GridSearchCV
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        cv = 5
        rf_gs = "random_forest_grid_search.pkl"
        train.train_random_forest_GridSearch(X_train, y_train, param_grid, cv, MODEL_DIR, rf_gs)

        mlflow.log_param(key="PROCESSED_DATA_PATH", value=PROCESSED_DATA_PATH)
        mlflow.log_param(key="MODEL_DIR", value=MODEL_DIR)
        mlflow.log_param(key="RF_RandomSearch_params", value=param_distribs)
        mlflow.log_param(key="RF_RandomSearch_n_iter", value=n_iter)
        mlflow.log_param(key="RF_GridSearch_params", value=param_grid)
        mlflow.log_artifact(XTRAIN_FILEPATH)
        mlflow.log_artifact(XTEST_FILEPATH)
        mlflow.log_artifact(YTRAIN_FILEPATH)
        mlflow.log_artifact(YTEST_FILEPATH)

    # Model Evaluation
    # =========================
    with mlflow.start_run(
        run_name="Child_HousingRun_Score", experiment_id=experiment_id, nested=True
    ):
        # Load Processed Data
        X_train, y_train, X_test, y_test = score.load_processed_data(PROCESSED_DATA_PATH)

        # Load Models
        model_lr = score.load_model(MODEL_DIR, lr)
        model_dt = score.load_model(MODEL_DIR, dt)
        model_rfrs = score.load_model(MODEL_DIR, rf_rs)
        model_rfgs = score.load_model(MODEL_DIR, rf_gs)

        train_columns_list = X_train.columns.tolist()
        # log Feature Importance of models
        score.get_feature_importance(model_lr, lr, train_columns_list)
        score.get_feature_importance(model_dt, dt, train_columns_list)
        score.get_feature_importance(model_rfrs, rf_rs, train_columns_list)
        score.get_feature_importance(model_rfgs, rf_gs, train_columns_list)

        # Scoring the models
        # Scoring LR
        lr_train_rmse, lr_test_rmse = score.score_model(
            model_lr, lr, X_train, y_train, X_test, y_test
        )
        # scoring DT
        dt_train_rmse, dt_test_rmse = score.score_model(
            model_dt, dt, X_train, y_train, X_test, y_test
        )
        # Scoring Random Forest (Tuned with RandomSearchCV)
        rfrs_train_rmse, rfrs_test_rmse = score.score_model(
            model_rfrs, rf_rs, X_train, y_train, X_test, y_test
        )
        # Scoring Random Forest (Tuned with GridSearchCV)
        rfgs_train_rmse, rfgs_test_rmse = score.score_model(
            model_rfgs, rf_gs, X_train, y_train, X_test, y_test
        )
        mlflow.log_metric(key="lr_train_rmse", value=lr_train_rmse)
        mlflow.log_metric(key="lr_test_rmse", value=lr_test_rmse)
        mlflow.log_metric(key="dt_train_rmse", value=dt_train_rmse)
        mlflow.log_metric(key="dt_test_rmse", value=dt_test_rmse)
        mlflow.log_metric(key="rfrs_train_rmse", value=rfrs_train_rmse)
        mlflow.log_metric(key="rfrs_test_rmse", value=rfrs_test_rmse)
        mlflow.log_metric(key="rfgs_train_rmse", value=rfgs_train_rmse)
        mlflow.log_metric(key="rfgs_test_rmse", value=rfgs_test_rmse)

        mlflow.sklearn.log_model(model_lr, "model_lr")
        mlflow.sklearn.log_model(model_dt, "model_dt")
        mlflow.sklearn.log_model(model_rfrs, "model_rfrs")
        mlflow.sklearn.log_model(model_rfgs, "model_rfgs")

        # All Results in Prettytable
        results = PrettyTable(["Model", "Train RMSE", "Test RMSE"])
        results.add_row(["Linear Regression", lr_train_rmse, lr_test_rmse])
        results.add_row(["Decision Tree", dt_train_rmse, dt_test_rmse])
        results.add_row(["RandomForest(Random Search)", rfrs_train_rmse, rfrs_test_rmse])
        results.add_row(["RandomForest(Grid Search)", rfgs_train_rmse, rfgs_test_rmse])
        logging.info(str(results))
        print(results)
