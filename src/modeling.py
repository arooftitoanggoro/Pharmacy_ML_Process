
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from datetime import datetime
from tqdm import tqdm
import yaml
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import hashlib
import util as util

def time_stamp():
    return datetime.now()

def create_log_template():
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
        "data_configurations" : [],
    }
    return logger

def training_log_updater(current_log, log_path):
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log

def train_eval_model(list_of_model, prefix_model_name, x_train, y_train, data_configuration_name, x_valid, y_valid, log_path):

    list_of_model = copy.deepcopy(list_of_model)
    logger = create_log_template()

    for model in tqdm(list_of_model):    
        model_name = prefix_model_name + "-" + model["model_name"]

        start_time = time_stamp()
        model["model_object"].fit(x_train, y_train)
        finished_time = time_stamp()

        elapsed_time = finished_time - start_time
        elapsed_time = elapsed_time.total_seconds()

        y_pred = model["model_object"].predict(x_valid)
        performance = mean_absolute_error(y_valid, y_pred)

        plain_id = str(start_time) + str(finished_time)
        chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

        model["model_uid"] = chiper_id

        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["performance"].append(performance)
        logger["data_configurations"].append(data_configuration_name)

    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model

def training_log_to_df(training_log):
    training_res = pd.DataFrame(training_log)
    training_res = training_res.explode(['model_name', 'model_uid', 'training_time', 'training_date', 'performance', 'data_configurations'])
   
    training_res.sort_values(["performance", "training_time"], ascending = [False, True], inplace = True)
    training_res.reset_index(inplace = True, drop = True)
    
    return training_res

def get_best_model(training_log_df, list_of_model):
    model_object = None

    best_model_info = training_log_df.sort_values(["performance", "training_time"], ascending = [False, True]).iloc[0]
    
    for model_data in list_of_model:
            if model_data["model_uid"] == best_model_info["model_uid"]:
                model_object = model_data["model_object"]
                break
    
    if model_object == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    return model_object

if __name__ == "__main__":
    # 1. Load configuration file
    params = util.load_config()

    # 2. Load Dataset
    x_train = joblib.load("data/processed/x_train_feng.pkl")
    y_train = joblib.load("data/processed/y_train_feng.pkl")

    x_valid = joblib.load("data/processed/x_valid_feng.pkl")
    y_valid = joblib.load("data/processed/y_valid_feng.pkl")

    x_test = joblib.load("data/processed/x_test_feng.pkl")
    y_test = joblib.load("data/processed/y_test_feng.pkl")

    # 3. Training Baseline Model
    dct_baseline = DecisionTreeRegressor()
    rfc_baseline = RandomForestRegressor()

    list_of_model = {
        "baseline": [
            { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
            { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""}
            ]
    }

    training_log, list_of_model_bs = train_eval_model(
        list_of_model["baseline"],
        "baseline_model",
        x_train,
        y_train,
        "baseline",
        x_valid,
        y_valid,
        "log/training_log.json"
    )

    list_of_model ['baseline']= copy.deepcopy(list_of_model_bs)

    # 4. Choose best performance baseline model
    training_res = training_log_to_df(training_log)
    model = get_best_model(training_res, list_of_model_bs)
    joblib.dump(model, "models/model.pkl")

    # 5. Hyperparameter Tuning
    dist_params_rfc = {
        "criterion" : ["mse"],
        "n_estimators" : [5, 10, 30, 50, 100, 500],
        "min_samples_split" : [1, 2, 4, 6, 10, 15, 20, 25],
        "min_samples_leaf" : [1, 2, 4, 6, 10, 15, 20, 25]
    }
    
    rfc_enh = GridSearchCV(RandomForestRegressor(), dist_params_rfc, n_jobs = -1, verbose = 420)
    list_of_model["hyperparams"] = [    {
        "model_name": rfc_enh.__class__.__name__ + "-" + rfc_enh.estimator.__class__.__name__,      
        "model_object": copy.deepcopy(rfc_enh),      
        "model_uid": ""}]

    training_log, list_of_model_hyp = train_eval_model(
        [list_of_model["hyperparams"][0]],
        "hyperparams",
        x_train,
        y_train,
        "hyperparams",
        x_valid,
        y_valid,
        "log/training_log.json"
    )

    list_of_model["hyperparams"][-1]= copy.deepcopy(list_of_model_hyp[0])
    
    # 6. Choose best performance hyperparameter model
    training_res_hyp = training_log_to_df(training_log)
    model = get_best_model(training_res_hyp, list_of_model_hyp)
    joblib.dump(model, "models/production_model.pkl")



