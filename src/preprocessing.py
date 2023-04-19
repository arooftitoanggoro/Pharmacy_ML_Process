import pandas as pd
import numpy as np
import seaborn as sns
import util as util
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set

def join_label_categori(set_data, config_data):
    # Check if label not found in set data
    if config_data["label"] in set_data.columns.to_list():
        # Create copy of set data
        set_data = set_data.copy()

        # Return renamed set data
        return set_data
    else:
        raise RuntimeError("Kolom label tidak terdeteksi pada set data yang diberikan!")
    
def categorical_nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace NaN with "UNKNOWN" in categorical columns
    categorical_columns = set_data.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        set_data[column] = set_data[column].fillna("UNKNOWN")

    # Return replaced set data
    return set_data

def numereical_nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace NaN with median in numerical columns
    numerical_columns = set_data.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        if set_data[column].isnull().sum() > 0:
            median = set_data[column].median()
            set_data[column].fillna(median, inplace=True)

    # Return replaced set data
    return set_data

def le_fit_transform(data_sets: list, config_data: dict) -> list:
    # Select categorical columns
    categorical_cols = data_sets[0].select_dtypes(include=['object']).columns.tolist()

    # Create le objects
    le_encoders = {col: LabelEncoder() for col in categorical_cols}

    # Fit and transform le for each categorical column in all data sets
    for col in categorical_cols:
        all_data = pd.concat([ds[col] for ds in data_sets], axis=0)
        le_encoders[col].fit(all_data)
        for ds in data_sets:
            ds[col] = le_encoders[col].transform(ds[col])

    # Save le objects
    util.pickle_dump(le_encoders, config_data["le_encoder_path"])

    # Return transformed data sets
    return data_sets

def le_transform(data: pd.DataFrame, config_data: dict) -> pd.DataFrame:
    # Load le object
    le_encoders = util.pickle_load(config_data["le_encoder_path"])

    # Find categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Transform categorical columns with le
    for col in categorical_cols:
        data[col] = le_encoders[col].transform(data[col])

    # Return transformed data
    return data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Join label categories
    train_set = join_label_categori(
        train_set,
        config_data
    )
    valid_set = join_label_categori(
        valid_set,
        config_data
    )
    test_set = join_label_categori(
        test_set,
        config_data
    )

    # 4. Handling Missing Value
    train_set = categorical_nan_detector(train_set)
    train_set = numereical_nan_detector(train_set)
    valid_set = categorical_nan_detector(train_set)
    valid_set = numereical_nan_detector(train_set)
    test_set = categorical_nan_detector(train_set)
    test_set = numereical_nan_detector(train_set)

    # 5. Encoding Categorical
    train_set, valid_set, test_set = le_fit_transform([train_set, valid_set, test_set], config_data)
    train_set = le_transform(train_set, config_data)
    valid_set = le_transform(valid_set, config_data)
    test_set = le_transform(test_set, config_data)

    # 6. Dumping Dataset
    x_train = train_set.drop(columns = "Profit")
    y_train = train_set.Profit

    util.pickle_dump(x_train, "data/processed/x_train_feng.pkl")
    util.pickle_dump(y_train, "data/processed/y_train_feng.pkl")

    util.pickle_dump(valid_set.drop(columns = "Profit"), "data/processed/x_valid_feng.pkl")
    util.pickle_dump(valid_set.Profit, "data/processed/y_valid_feng.pkl")

    util.pickle_dump(test_set.drop(columns = "Profit"), "data/processed/x_test_feng.pkl")
    util.pickle_dump(test_set.Profit, "data/processed/y_test_feng.pkl")


