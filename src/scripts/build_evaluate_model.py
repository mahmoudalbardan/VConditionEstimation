import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils import get_config, parse_args

sns.set_theme(style="whitegrid")


def read_file(config):
    """
    Read a CSV file from GCP storage bucket.
    This function connects to Google Cloud Storage, extract the specified CSV file,
    "credictcard.csv" and loads it into a Pandas DataFrame.

    Parameters
    ----------
    gcs_bucket_name : str
        The name of the Google Cloud Storage bucket.
    gcs_filename : str
        The name of the file within the bucket.

    Returns
    -------
    pd.DataFrame
        The contents of "credictcard.csv".
    """
    data_fs1 = pd.read_csv(config["FILES"]["FS"], sep="\t",header=None)
    data_ps2 = pd.read_csv(config["FILES"]["PS"], sep="\t",header=None)
    data_profile = pd.read_csv(config["FILES"]["PROFILE"],sep="\t", header=None)
    return data_fs1, data_ps2, data_profile


def compute_features_one_cycle(cycle_data_fs1, cycle_data_ps2):
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
    Fit an Isolation Forest model to the transformed data.
    This function preprocesses the feature data by scaling it using
    StandardScaler, then fits an Isolation Forest model.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        The transformed input data containing features and the target variable.

    Returns
    -------
    IsolationForest
        The fitted Isolation Forest model.
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
    Save the trained model to a specified file path.
    This function uses joblib to save the fitted model.

    Parameters
    ----------
    model : IsolationForest
        The fitted Isolation Forest model.

    model_path : str
        The model file path it will be saved.
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
