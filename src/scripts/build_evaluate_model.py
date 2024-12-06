import argparse
import configparser
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


def parse_args():
    """
    Parse command line arguments.
    This function sets up the argument parser to handle command line
    input for the configuration file and retrain flag. It defines
    the expected arguments and their types.

    Returns
    -------
    Namespace
        A Namespace object containing the parsed command line arguments.
        - configuration : str
            Path to the configuration file (default: 'configuration.ini').
        - eda : str
            Flag indicating whether to do exploratory data analysis (plot histograms) ('true' or 'false'; default: 'false').
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=str,
                        help='configuration file', default='configuration.ini')
    parser.add_argument("--eda", type=str,
                        help='true or false: true corresponds to a EDA ',
                        default='false')

    args = parser.parse_args()
    return args


def get_config(configfile):
    """
    Read a configuration file.
    This function reads the specified configuration file using the
    configparser module and returns the configuration object.

    Parameters
    ----------
    configfile : str
        The path to the configuration file to read.

    Returns
    -------
    config : configparser.ConfigParser
        A ConfigParser object containing the configuration settings
        loaded from the specified file.
    """
    config = configparser.ConfigParser()
    config.read(configfile)
    return config


def read_file(config):
    """
    Read a files

    Parameters
    ----------
    config : configparser.ConfigParser
        A ConfigParser object containing the configuration settings
        loaded from the specified file.

    Returns
    -------
    pd.DataFrame
        The contents of three files in three dataframes
    """
    data_fs1 = pd.read_csv(config["FILES"]["FS"], sep="\t", header=None)
    data_ps2 = pd.read_csv(config["FILES"]["PS"], sep="\t", header=None)
    data_profile = pd.read_csv(config["FILES"]["PROFILE"], sep="\t", header=None)
    return data_fs1, data_ps2, data_profile


def compute_features_one_cycle(cycle_data_fs1, cycle_data_ps2):
    """
    Compute statistical features for two sets of cycle data (FS1 and PS2) from a hydraulic system cycle.

    This function computes several descriptive statistics for each of the input cycle data (FS1 and PS2),
    including mean, max, min, standard deviation, percentiles (25th, 50th, 75th),
    as well as the count of peaks and troughs in each dataset.

    Parameters:
    ----------
    cycle_data_fs1 (array-like): A numeric array or list containing the FS1 cycle data.
    cycle_data_ps2 (array-like): A numeric array or list containing the PS2 cycle data.

    Returns:
    ----------
    dict: A dictionary containing the following statistical features:
        - 'FS1_mean': Mean of FS1 cycle data
        - 'FS1_max': Maximum value of FS1 cycle data
        - 'FS1_min': Minimum value of FS1 cycle data
        - 'FS1_std': Standard deviation of FS1 cycle data
        - 'FS1_25th': 25th percentile of FS1 cycle data
        - 'FS1_50th': 50th percentile (median) of FS1 cycle data
        - 'FS1_75th': 75th percentile of FS1 cycle data
        - 'PS2_mean': Mean of PS2 cycle data
        - 'PS2_max': Maximum value of PS2 cycle data
        - 'PS2_min': Minimum value of PS2 cycle data
        - 'PS2_std': Standard deviation of PS2 cycle data
        - 'PS2_25th': 25th percentile of PS2 cycle data
        - 'PS2_50th': 50th percentile (median) of PS2 cycle data
        - 'PS2_75th': 75th percentile of PS2 cycle data
        - 'FS1_peak_count': Number of peaks in FS1 cycle data
        - 'FS1_trough_count': Number of troughs in FS1 cycle data
        - 'PS2_peak_count': Number of peaks in PS2 cycle data
        - 'PS2_trough_count': Number of troughs in PS2 cycle data

    Notes:
    - The function uses `find_peaks` from scipy to detect peaks and troughs in the cycle data.
    - The input cycle data should be numeric arrays or lists.
    """
    features = {}

    # FS1
    features['FS1_mean'] = np.mean(cycle_data_fs1)
    features['FS1_max'] = np.max(cycle_data_fs1)
    features['FS1_min'] = np.min(cycle_data_fs1)
    features['FS1_std'] = np.std(cycle_data_fs1)
    features['FS1_25th'] = np.percentile(cycle_data_fs1, 25)
    features['FS1_50th'] = np.percentile(cycle_data_fs1, 50)
    features['FS1_75th'] = np.percentile(cycle_data_fs1, 75)

    # PS2
    features['PS2_mean'] = np.mean(cycle_data_ps2)
    features['PS2_max'] = np.max(cycle_data_ps2)
    features['PS2_min'] = np.min(cycle_data_ps2)
    features['PS2_std'] = np.std(cycle_data_ps2)
    features['PS2_25th'] = np.percentile(cycle_data_ps2, 25)
    features['PS2_50th'] = np.percentile(cycle_data_ps2, 50)  # Median
    features['PS2_75th'] = np.percentile(cycle_data_ps2, 75)

    # Peaks and Troughs for FS1
    peaks_fs1, _ = find_peaks(cycle_data_fs1)
    troughs_fs1, _ = find_peaks(-cycle_data_fs1)
    features['FS1_peak_count'] = len(peaks_fs1)
    features['FS1_trough_count'] = len(troughs_fs1)

    # Peaks and Troughs for PS2
    peaks_ps2, _ = find_peaks(cycle_data_ps2)
    troughs_ps2, _ = find_peaks(-cycle_data_ps2)
    features['PS2_peak_count'] = len(peaks_ps2)
    features['PS2_trough_count'] = len(troughs_ps2)
    return features


def compute_features(data_fs1, data_ps2):
    all_features = []
    for j in range(len(data_fs1)):
        cycle_data_fs1 = np.array(data_fs1.loc[j, :])
        cycle_data_ps2 = np.array(data_ps2.loc[j, :])
        features = compute_features_one_cycle(cycle_data_fs1, cycle_data_ps2)
        all_features.append(features)
    df_features = pd.DataFrame(all_features)
    return df_features


def process_data(data_fs1, data_ps2, data_profile):
    """
    Process the cycle data to compute features, prepare the target variable, and split the data into training and testing sets.

    Parameters
    ----------
    data_fs1 : array-like
        A numeric array or list containing the FS1 cycle data.
    data_ps2 : array-like
        A numeric array or list containing the PS2 cycle data.
    data_profile : array-like or DataFrame
        A dataset containing the target variable, with the target values in the second column.

    Returns
    -------
    tuple
        A tuple containing two pandas DataFrames:
        - data_train : DataFrame
            The training dataset with features and binary target variable.
        - data_test : DataFrame
            The testing dataset with features and binary target variable.

    Notes
    -----
    - The function assumes `data_profile[1]` contains the target values, which are used to classify the target as 1 for '100' and 0 otherwise.
    - The dataset is split into training and testing sets with the first 2000 rows for training and the rest for testing.
    """

    df_features = compute_features(data_fs1, data_ps2)
    df_target = pd.DataFrame(data_profile[1])
    df_target.columns = ["target"]
    data = pd.concat([df_features, df_target], axis=1)
    data["target"] = data["target"].apply(lambda x: 1 if x == 100 else 0)
    data_train, data_test = data.iloc[:2000], data.iloc[2000:]
    return data_train, data_test


def explore_data(eda, data):
    """
    Perform exploratory data analysis on the dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data transactions.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.DataFrame: Info summary of the DataFrame.
        - pd.DataFrame: Descriptive statistics of the DataFrame.
        - pd.Series: Frequency count of each class in the 'Class' column.
    """
    if eda == "true":
        data_info = data.info()
        data_describe = data.describe()
        data_target_frequency = data['target'].value_counts()
        features = data.columns[:-1]

        # Histogram plots (univariate analysis)
        figure, axes = plt.subplots(3, 7, figsize=(50, 30))
        axes = axes.flatten()
        sns.set_context(font_scale=0.8)
        for j, feature in enumerate(features):
            sns.histplot(data[feature].values, bins=50, color="c", kde=True, ax=axes[j])
            axes[j].set_xlabel(feature)

        plt.show()
        print(data_info, data_describe, data_target_frequency)


def transform_data(data):
    """
    Transform features in the DataFrame to reduce skewness.

    This function applies a log transformation to features in the DataFrame
    that have a skewness greater than 2.

    Parameters
    ----------
    data : pd.DataFrame
        The original dataframe.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with transformed features.
    """
    data_transformed = data.copy()
    features_to_log = []
    features = data.columns[:-1]
    for feature in features:
        if np.abs(data[feature].skew()) > 2:
            # Apply log(1+x) to avoid issues with log(0)
            data_transformed[feature] = np.sign(data[feature].values) * \
                                        np.log1p(np.abs(data[feature].values))
            features_to_log.append(feature)
    return data_transformed, features_to_log


def fit_validate_model(data_transformed):
    """
    Fit and validate a Random Forest classifier model using the transformed data.

    This function preprocesses the feature data by scaling it with StandardScaler,
    then fits a Random Forest classifier. It performs hyperparameter tuning using
    GridSearchCV and returns the best model, the scaler, and the best cross-validation score.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        The transformed input data containing features and the target variable.
        The target variable should be in the column named 'target'.

    Returns
    -------
    best_model : RandomForestClassifier
        The fitted Random Forest model with the best hyperparameters based on GridSearchCV.
    scaler : StandardScaler
        The StandardScaler object used to scale the features.
    best_score : float
        The best cross-validation score obtained during the hyperparameter tuning.

    Notes
    -----
    - This function uses `GridSearchCV` to perform hyperparameter tuning on the Random Forest model.
    - The hyperparameters being tuned are `n_estimators` and `max_depth`.
    """

    X = data_transformed.drop(columns=['target'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = data_transformed['target']

    # Define Random Forest Classifier
    model = RandomForestClassifier(random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [4, 7, 10],
    }

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=10,
                               scoring='accuracy')
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    best_model.fit(X_scaled, y)
    return best_model, scaler, grid_search.best_score_


def evaluate_model(model, scaler, features_to_log, data_test):
    for feature in features_to_log:
        data_test[feature] = np.sign(data_test[feature].values) * \
                             np.log1p(np.abs(data_test[feature].values))

    X = data_test.drop(columns=['target'])
    X_test = scaler.transform(X)
    y_true = data_test['target']
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_true, y_pred)
    return test_accuracy


def save_outputs(config, model, dict_metrics):
    """
    Save the trained model and performance metrics to specified file paths.

    This function saves the fitted model using `joblib` and the performance metrics
    to a JSON file. The file paths for saving the model and metrics are retrieved
    from the provided configuration.

    Parameters
    ----------
    config : dict
        A dictionary containing file paths for saving the model and metrics.
        It should contain keys `"FILES"`, with `"MODEL_PATH"` for the model and `"METRICS_PATH"` for the metrics.

    model : object
        The trained model to be saved, random forest classifier

    dict_metrics : dict
        A dictionary containing the performance metrics to be saved in JSON format.

    Notes
    -----
    - The model is saved using `joblib.dump()`.
    - The metrics are saved in a JSON format using `json.dump()`, with an indentation level of 4.
    """
    joblib.dump(model, config["FILES"]["MODEL_PATH"])
    with open(config["FILES"]["METRICS_PATH"], "w") as json_file:
        json.dump(dict_metrics, json_file, indent=4)


def main(args):
    """
    Main function

    Parameters
    ----------
    args : Namespace
        Parsed command line arguments containing configuration and retrain flag.
    """
    config = get_config(args.configuration)
    eda = args.eda
    data_fs1, data_ps2, data_profile = read_file(config)
    data_train, data_test = process_data(data_fs1, data_ps2, data_profile)
    explore_data(eda, data_train)
    data_transformed, features_to_log = transform_data(data_train)
    model, scaler, validation_accuracy = fit_validate_model(data_transformed)
    test_accuracy = evaluate_model(model, scaler, features_to_log, data_test)
    dict_metrics = {"validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy}
    save_outputs(config, model, dict_metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
