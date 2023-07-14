import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit


def fetch_housing_data(housing_url, housing_path):
    """This function downloads the housing data for this project.

    If the URL is incorrect or file not found at the mentioned URL then this function exits.

    Args:
        housing_url: housing data will we downloded using this 'url'.

        housing_path: Downloaded housing data will be saved at this path.
    Returns:
        None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    try:
        urllib.request.urlretrieve(housing_url, tgz_path)
    except Exception:
        logging.error("Invalid URL : " + str(housing_url))
        logging.error("or 'from six.moves import urllib' is not imported.")
        logging.error("Exiting..")
        exit()

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """This function loads and returns the housing data.

    Args:
        housing_path : path where the housing data was saved.
    Returns:
        Data as pandas dataframe loaded from housing_path.
    Raises:
        Exit the code if housing_path does not exists.
    """
    logging.info("load_housing_data")
    csv_path = os.path.join(housing_path, "housing.csv")
    if os.path.exists(csv_path) is False:
        logging.error("File " + str(csv_path) + " does not exist.")
        logging.error("Exiting...")
        exit()
    return pd.read_csv(csv_path)


def train_test_split(df, test_size=0.2, random_state=42):
    """This function returns the train and test data using StratifiedShuffleSplit method.

    Args:
        df: pandas dataframe

        test_size: value between 0 and 1 that represents the percentage of data as test data. \
            Default values is 0.2

        random_state: any number as seed for the purpose of reproducibility.
    Returns:
        train_set,test_set
    """
    housing = df.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set


def feature_engineering(
    train_set, test_set, XTRAIN_FILEPATH, YTRAIN_FILEPATH, XTEST_FILEPATH, YTEST_FILEPATH
):
    """Saves the feature engineered train and test data.

    This function performs the feature engineering for housing data and
    saves the final train and test data at specified path.

    Args:
        train_set : housing train dataframe

        test_set : housing test dataframe

        XTRAIN_FILEPATH : path to save X_train

        YTRAIN_FILEPATH : path to save y_train

        XTEST_FILEPATH : path to save X_test

        YTEST_FILEPATH : path to save y_test
    Returns:
        None

    """
    housing = train_set.copy()
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)  # drop labels for training set

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    y_test = test_set["median_house_value"].copy()
    X_test = test_set.drop("median_house_value", axis=1)

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)

    X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    logging.info("Saving Data")
    housing_prepared.to_csv(XTRAIN_FILEPATH, index=False)
    housing_labels.to_csv(YTRAIN_FILEPATH, index=False)
    X_test_prepared.to_csv(XTEST_FILEPATH, index=False)
    y_test.to_csv(YTEST_FILEPATH, index=False)
