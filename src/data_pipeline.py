from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input, params, api = False):
    input = copy.deepcopy(input)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        #assert input.select_dtypes("datetime").columns.to_list() == \
            #params["datetime_columns"], "an error occurs in datetime column(s)."
        assert input.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input.select_dtypes("float").columns.to_list() == \
            params["float64_columns"], "an error occurs in float64 column(s)."
    else:
        # In case checking data from api
        # Predictor that has object dtype only stasiun
        object_columns = params["object_columns"]
        del object_columns[1:]

        # Max column not used as predictor
        float_columns = params["float64_columns"]
        del float_columns[-1]

        # Check data types
        assert input.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input.select_dtypes("float").columns.to_list() == \
            float_columns, "an error occurs in float64 column(s)."

    assert input.Quantity.between(params["range_Quantity"][0], params["range_Quantity"][1]).sum() == len(input), "an error occurs in Quantity range."
    assert input.Sales.between(params["range_Sales"][0], params["range_Sales"][1]).sum() == len(input), "an error occurs in Sales range."
    assert input.Discount.between(params["range_Discount"][0], params["range_Discount"][1]).sum() == len(input), "an error occurs in Discount range."
    assert input.Profit.between(params["range_Profit"][0], params["range_Profit"][1]).sum() == len(input), "an error occurs in Profit range."

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    #5. Remove some columns
    raw_dataset = raw_dataset.drop("Row ID", axis=1)
    raw_dataset = raw_dataset.drop("Order Date", axis=1)
    raw_dataset = raw_dataset.drop("Ship Date", axis=1)

    #6. Handle Variabel "Quantity"
    raw_dataset["Quantity"] = raw_dataset["Quantity"].astype("float64")

    #7. Load Pickle
    util.pickle_dump(
        raw_dataset,
        config_data["cleaned_raw_dataset_path"]
    )
    # 8. Check data definition
    check_data(raw_dataset, config_data)

    # 9. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.Profit.copy()

    # 10. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.3,
        random_state = 123
    )

    # 11. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.5,
        random_state = 123
    )

    # 16. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])