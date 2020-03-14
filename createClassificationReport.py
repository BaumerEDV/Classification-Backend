from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import \
    roc_auc_score  # https://elitedatascience.com/imbalanced-classes
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from statistics import median, mean, stdev
from joblib import dump
import numpy as np
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from SETTINGS import EXPORT_FEATURE_VECTOR_FILE_NAME, COMBINED_DATA_EXPORT_FILE_NAME, CLASSIFIER_JOBLIB_FILE_NAME, \
    SCALER_JOBLIB_FILE_NAME, DBM_NA_FILL_VALUE, CONFIRMATION_MEASUREMENT_EXPORT_FILE_NAME
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
import os
import warnings

# this code uses the word test dataset for the dataset that assesses the performance during training
# this code uses the word validation dataset for the dataset that assesses the real world generalizability of a model


K_FOLD_NUMBER = 5
RANDOMIZED_SEARCH_ITERATIONS = 10
SKIP_SEARCH = True
OVERSAMPLE_VALIDATION_DATASET = False
OVERSAMPLE_KFOLD_VALIDATION_DATASETS = False
KFOLD_TEST_UPPER_STORIES_ONLY = False
SCALERS = [
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
    preprocessing.RobustScaler,
    preprocessing.MaxAbsScaler
]
SCALER = SCALERS[0]
OVERSAMPLE_ALL_CLASSIFIERS = False
SHOW_VALIDATION_SET_PERFORMANCE = False
SHOW_KFOLD_CROSS_VALIDATION_PERFORMANCE = False
SHOW_CONFIRMATION_SET_PERFORMANCE = True
SUPPRESS_ALL_WARNINGS = False

CLASSIFIERS_WITH_HYPERPARAMETER_DISTRIBUTIONS = [
    # {
    #    "name": "K(3) Nearest Neighbor",
    #    "classifier": Pipeline([
    #        # ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", KNeighborsClassifier())
    #    ]),
    #    "param_dist": {
    #        "classification__n_neighbors": [3, 4, 5, 6],
    #        "classification__weights": ["uniform", "distance"],
    #        "classification__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    #        "classification__leaf_size": [10, 20, 30, 75, 150, 300],
    #        "classification__p": [1, 2],
    #        "classification__n_jobs": [-1]
    #    },
    #    "iterations": 250
    # },
    # {
    #    "name": "Feed Forward Neural Network",
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", MLPClassifier())
    #    ]),
    #    "param_dist": {
    #        "classification__hidden_layer_sizes": [50, 75, 100, 125, 140, 300, (100, 100)],
    #        "classification__activation": ["identity", "logistic", "tanh", "relu"],
    #        "classification__solver": ["lbfgs", "sgd", "adam"],
    #        "classification__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    #        "classification__learning_rate": ["constant", "invscaling", "adaptive"],
    #        "classification__learning_rate_init": [1e-4, 1e-3, 1e-2],
    #        "classification__max_iter": [500, 700, 1000],
    #        "classification__random_state": [42],
    #        "classification__momentum": stats.uniform(0, 1),
    #        "classification__nesterovs_momentum": [False, True],
    #        "classification__beta_1": [0.9999, 0.9, 0.8, 0.5, 0.1],
    #        "classification__beta_2": [0.9999, 0.999, 0.99, 0.9],
    #    },
    #    "iterations": 1000
    # },
    # {
    #    "name": "Random Forest Classifier",
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", RandomForestClassifier())
    #    ]),
    #    "param_dist": {
    #        "classification__n_estimators": [10, 50, 100, 125, 150, 175, 200, 250, 300],
    #        "classification__criterion": ["gini", "entropy"],
    #        "classification__max_features": ["sqrt", "log2", "auto"],
    #        "classification__n_jobs": [-1],
    #        "classification__random_state": [42],
    #        "classification__class_weight": [None, "balanced"]
    #    },
    #    "iterations": 120
    # },
    # {
    #    "name": "Multinomial Naive Bayes",
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", MultinomialNB())
    #    ]),
    #    "param_dist": {
    #        "classification__alpha": stats.uniform(0, 1),
    #        "classification__fit_prior": [False, True],
    #        "classification__class_prior": [None, [1 / 32] * 32]  # 32 is the number of classes
    #    },
    #    "iterations": 100
    # },
    {
        "name": "SVC",  # equal with pipeline
        "classifier": Pipeline([
            ("classification", SVC())
        ]),
        "param_dist": {
            "classification__random_state": [42],
            "classification__C": stats.uniform(0, 2),
            "classification__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "classification__degree": [2, 3, 4, 15, 30, 50, 100],
            "classification__gamma": ["scale", "auto"],
            "classification__coef0": stats.uniform(-5, 5),
            "classification__shrinking": [False, True],
            "classification__decision_function_shape": ["ovo", "ovr"],
        },
        "iterations": 50,
    },
    {
        "name": "AdaBoost",  # equal with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
                                                  n_estimators=200, random_state=42))
        ]),
        "param_dist": {
            "classification__random_state": [42],
            "classification__base_estimator": [DecisionTreeClassifier(max_depth=1, random_state=42),
                                               DecisionTreeClassifier(max_depth=2, random_state=42),
                                               DecisionTreeClassifier(max_depth=3, random_state=42),
                                               DecisionTreeClassifier(max_depth=4, random_state=42),
                                               DecisionTreeClassifier(max_depth=5, random_state=42),
                                               DecisionTreeClassifier(max_depth=6, random_state=42),
                                               DecisionTreeClassifier(max_depth=7, random_state=42),
                                               DecisionTreeClassifier(max_depth=8, random_state=42),
                                               DecisionTreeClassifier(max_depth=9, random_state=42),
                                               DecisionTreeClassifier(max_depth=10, random_state=42),
                                               DecisionTreeClassifier(max_depth=11, random_state=42),
                                               DecisionTreeClassifier(max_depth=12, random_state=42),
                                               DecisionTreeClassifier(max_depth=13, random_state=42),
                                               DecisionTreeClassifier(max_depth=14, random_state=42),
                                               DecisionTreeClassifier(max_depth=15, random_state=42)],
            "classification__n_estimators": [30, 50, 70, 80, 100, 150, 200, 250, 300, 400, 500],
            "classification__learning_rate": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.6, 2, 5],
            "classification__algorithm": ["SAMME", "SAMME.R"]
        },
        "iterations": 100
    },
    {
        "name": "Gradient Boosting Classifier",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            (
                "classification",
                GradientBoostingClassifier(random_state=42, n_estimators=400, max_depth=5, subsample=0.5))
        ]),
        "iterations": 100,
        "param_dist": {
            "classification__random_state": [42],
            "classification__n_estimators": [30, 50, 70, 80, 100, 150, 200, 250, 300, 400, 500],
            "classification__max_depth": [1, 2, 3, 4, 5, 10, 15, 20],
            "classification__subsample": stats.uniform(0, 1),
            "classification__learning_rate": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.6, 2, 5],
            "classification__max_features": ["sqrt", "log2", "None"],
        }
    },
    {
        "name": "Bagging Classifier",  # significantly better with pipeline - ties for best classifier
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", BaggingClassifier())
        ]),
        "iterations": 200,
        "param_dist": {
            "classification__random_state": [42],
            "classification__n_estimators": [30, 50, 70, 80, 100, 150, 200, 250, 300, 400, 500],
            "classification__n_jobs": [-1],
            "classification__base_estimator": [DecisionTreeClassifier(max_depth=1, random_state=42),
                                               DecisionTreeClassifier(max_depth=2, random_state=42),
                                               DecisionTreeClassifier(max_depth=3, random_state=42),
                                               DecisionTreeClassifier(max_depth=4, random_state=42),
                                               DecisionTreeClassifier(max_depth=5, random_state=42),
                                               DecisionTreeClassifier(max_depth=6, random_state=42),
                                               DecisionTreeClassifier(max_depth=7, random_state=42),
                                               DecisionTreeClassifier(max_depth=8, random_state=42),
                                               DecisionTreeClassifier(max_depth=9, random_state=42),
                                               DecisionTreeClassifier(max_depth=10, random_state=42),
                                               DecisionTreeClassifier(max_depth=11, random_state=42),
                                               DecisionTreeClassifier(max_depth=12, random_state=42),
                                               DecisionTreeClassifier(max_depth=13, random_state=42),
                                               DecisionTreeClassifier(max_depth=14, random_state=42),
                                               DecisionTreeClassifier(max_depth=15, random_state=42),
                                               DecisionTreeClassifier(max_depth=16, random_state=42),
                                               DecisionTreeClassifier(max_depth=17, random_state=42),
                                               DecisionTreeClassifier(max_depth=18, random_state=42)],
            "classification__oob_score": [True, False],
        }
    },
]
CLASSIFIERS_FOR_EVALUATION = [
    {
        "name": "K Nearest Neighbor",  # worse with pipeline
        "classifier": KNeighborsClassifier(weights='distance', p=1, n_neighbors=3,
                                           n_jobs=-1, leaf_size=10, algorithm='auto'),
    },
    {
        "name": "Feed Forward Neural Network",  # worse with pipeline
        "classifier": MLPClassifier(activation="logistic", alpha=0.01, beta_1=0.9, beta_2=0.999,
                                    hidden_layer_sizes=300, learning_rate="constant", learning_rate_init=0.001,
                                    max_iter=700, momentum=0.7206236910652511, nesterovs_momentum=False,
                                    random_state=42, solver="adam"),
    },
    {
        "name": "Multinomial Naive Bayes",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", MultinomialNB(alpha=0.11678761101807189, class_prior=[1 / 32] * 32, fit_prior=False))
            # 32 is the number of classes, this must be changed if number of classes changes
        ]),
    },
    {
        "name": "Random Forest",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='log2',
                                                      criterion='gini', class_weight=None))
        ])
    },
    {
        "name": "SVC",  # equal with pipeline
        "classifier": SVC(kernel="linear", C=0.5533914322723179, coef0=-1.5082623301547131,
                          decision_function_shape="ovo", degree=3, gamma="auto", shrinking=False, probability=True,
                          random_state=42)
    },
    {
        "name": "AdaBoost Classifier",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=0.9,
                                                  base_estimator=DecisionTreeClassifier(ccp_alpha=0.0,
                                                                                        class_weight=None,
                                                                                        criterion="gini", max_depth=12,
                                                                                        max_features=None,
                                                                                        max_leaf_nodes=None,
                                                                                        min_impurity_decrease=0.0,
                                                                                        min_impurity_split=None,
                                                                                        min_samples_leaf=1,
                                                                                        min_samples_split=2,
                                                                                        min_weight_fraction_leaf=0.0,
                                                                                        presort="deprecated",
                                                                                        random_state=42,
                                                                                        splitter="best"),
                                                  algorithm="SAMME.R"))
        ])
    },
    {
        "name": "Gradient Boosting Classifier",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            (
                "classification",
                GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features="sqrt", n_estimators=400,
                                           random_state=42, subsample=0.21503475283023676))
        ])
    },
    {
        "name": "Bagging Classifier",  # significantly better with pipeline - ties for best classifier
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", BaggingClassifier(random_state=42, oob_score=False, n_jobs=-1, n_estimators=200,
                                                 base_estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                                                       criterion="gini", max_depth=18,
                                                                                       max_features=None,
                                                                                       max_leaf_nodes=None,
                                                                                       min_impurity_decrease=0.0,
                                                                                       min_impurity_split=None,
                                                                                       min_samples_leaf=1,
                                                                                       min_samples_split=2,
                                                                                       min_weight_fraction_leaf=0.0,
                                                                                       presort="deprecated",
                                                                                       random_state=42,
                                                                                       splitter="best")))
        ])
    },  # maybe try a voting classifier?
]
MODEL_TO_SERIALIZE = Pipeline([
    ("sampling", RandomOverSampler(random_state=42)),
    ("classification", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='log2',
                                              criterion='gini', class_weight=None))
])


# from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report_search_top_results(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def save_feature_array_column_order(feature_array):
    feature_array[[False] * len(feature_array)].to_csv(
        EXPORT_FEATURE_VECTOR_FILE_NAME, index=False)


def assert_train_and_test_feature_vectors_are_fully_distinct(train_df, test_df):
    assert (pd.merge(train_df, test_df, on=train_df.columns.to_list(), how='outer',
                     indicator='Exist')['Exist'].value_counts().both == 0)


def oversample_df(df):
    # https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
    most_common_class = df['nodeId'].value_counts().idxmax()
    n_most_common_class = df['nodeId'].value_counts().max()
    base_df = df.loc[df['nodeId'] == most_common_class]
    new_df = pd.DataFrame()
    for class_name in df['nodeId'].unique():
        if class_name == most_common_class:
            new_df = pd.concat([new_df, base_df], axis=0)
            continue
        n_class = (df.nodeId == class_name).sum()
        class_oversampled_df = df.loc[df['nodeId'] == class_name].sample(n_most_common_class, replace=True,
                                                                         random_state=42)
        new_df = pd.concat([new_df, class_oversampled_df], axis=0)
    return new_df


def fit_model_on_test_and_validation_df(model, test_df, validation_df):
    complete_df = pd.concat([test_df, validation_df], axis=0)
    complete_target_df = complete_df['nodeId']
    complete_df_train = complete_df.drop(['nodeId'], axis=1)
    model.fit(complete_df_train, complete_target_df)
    return model


def save_model_and_scaler(model, scaler):
    dump(model, CLASSIFIER_JOBLIB_FILE_NAME)
    dump(scaler, SCALER_JOBLIB_FILE_NAME)


def create_classifier_pipline_that_oversamples_training_data(classifier):
    return Pipeline([
        ("sampling", RandomOverSampler(random_state=42)),
        ("classification", classifier)
    ])


def test_classifier_performance(classifier, X_train, y_train, X_validation, y_validation):
    if OVERSAMPLE_ALL_CLASSIFIERS:
        classifier = create_classifier_pipline_that_oversamples_training_data(classifier)
    classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_validation)
    y_predictions_proba = classifier.predict_proba(X_validation)
    performance = {
        "accuracy": accuracy_score(y_validation, y_predictions),
        "balanced accuracy": balanced_accuracy_score(y_validation, y_predictions),
        # "f1_score": f1_score(y_validation, y_predictions, average="weighted"),
        # "precision": precision_score(y_validation, y_predictions, average="weighted", zero_division=1),
        # "recall": recall_score(y_validation, y_predictions, average="weighted")
    }
    if not KFOLD_TEST_UPPER_STORIES_ONLY:
        # performance["roc_auc"] = roc_auc_score(y_validation, y_predictions_proba, multi_class="ovo", average="weighted")
        pass
    return performance


def conduct_random_search(X_train, y_train):
    for classifier_data in CLASSIFIERS_WITH_HYPERPARAMETER_DISTRIBUTIONS:
        iterations = classifier_data["iterations"]
        model = classifier_data["classifier"]
        param_dist = classifier_data["param_dist"]
        random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                           n_iter=iterations,
                                           cv=K_FOLD_NUMBER
                                           , scoring=make_scorer(balanced_accuracy_score)
                                           )
        random_search.fit(X_train, y_train)
        print(model)
        report_search_top_results(random_search.cv_results_)


def report_classifiers_performance_on_validation_set(X_train, y_train, X_validation, y_validation):
    for classifier_data in CLASSIFIERS_FOR_EVALUATION:
        performance = test_classifier_performance(classifier_data["classifier"], X_train, y_train,
                                                  X_validation, y_validation)
        print(classifier_data["name"])
        for key, value in performance.items():
            print(key, value)
        print("")


def load_and_clean_combined_data_df_from_file_name(filename):
    df = pd.read_csv(filename)
    df.fillna(DBM_NA_FILL_VALUE, inplace=True)
    df.drop(['timestamp', 'Unnamed: 0', 'pressure'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def get_scaled_data_and_scaler(df):
    scaler = SCALER()
    df[df.columns.difference(['nodeId'])] = scaler.fit_transform(
        df[df.columns.difference(['nodeId'])])
    return df, scaler


def split_into_train_and_validation_df(df):
    return train_test_split(df, test_size=0.33, random_state=36,
                            stratify=df['nodeId'])


def get_X_and_y_for_df(df):
    return df.drop(['nodeId'], axis=1), df['nodeId']


def get_X_and_y_for_train_and_validation(train_df, validation_df):
    X_train, y_train = get_X_and_y_for_df(train_df)
    X_validation, y_validation = get_X_and_y_for_df(validation_df)
    return X_train, y_train, X_validation, y_validation


def oversample_df_if_enabled(df):
    if OVERSAMPLE_VALIDATION_DATASET:
        return oversample_df(df)
    else:
        return df


def conduct_random_search_if_enabled(X_train, y_train):
    if not SKIP_SEARCH:
        conduct_random_search(X_train, y_train)


def report_classifiers_performance_on_stratified_kfold(master_df):
    for classifier_data in CLASSIFIERS_FOR_EVALUATION:
        skf = StratifiedKFold(n_splits=K_FOLD_NUMBER, random_state=42, shuffle=True)
        total_results = {}
        is_first_iteration = True
        X, y = get_X_and_y_for_df(master_df)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if KFOLD_TEST_UPPER_STORIES_ONLY:
                test_df = pd.merge(X_test, y_test, how="outer", left_index=True, right_index=True)
                filter = test_df['nodeId'].str.contains("(?:.*\.2\..*|.*\.3\..*)")
                test_df = test_df[filter]
                X_test, y_test = get_X_and_y_for_df(test_df)
            if OVERSAMPLE_KFOLD_VALIDATION_DATASETS:
                test_df = pd.merge(X_test, y_test, how="outer", left_index=True, right_index=True)
                test_df = oversample_df(test_df)
                X_test, y_test = get_X_and_y_for_df(test_df)
            results = test_classifier_performance(classifier_data["classifier"], X_train, y_train, X_test, y_test)
            for key, value in results.items():
                if is_first_iteration:
                    total_results[key] = []
                total_results[key].append(value)
            is_first_iteration = False
        print(classifier_data["name"])
        for key, value in total_results.items():
            print("Mean", key, mean(value))
        print("")


def get_scaled_df_from_scaler(df, scaler):
    df[df.columns.difference(['nodeId'])] = scaler.transform(
        df[df.columns.difference(['nodeId'])])
    return df


def load_confirmation_df_into_feature_vectors_using_scaler_based_on_master_df_classes(scaler, classes):
    confirmation_df = load_and_clean_combined_data_df_from_file_name(CONFIRMATION_MEASUREMENT_EXPORT_FILE_NAME)
    features_head = pd.read_csv(EXPORT_FEATURE_VECTOR_FILE_NAME)
    features_head['nodeId'] = {}
    confirmation_df = pd.concat(
        [features_head, confirmation_df[features_head.columns.intersection(confirmation_df.columns)]], sort=False)
    confirmation_df = confirmation_df[confirmation_df["nodeId"].isin(classes)]
    confirmation_df.drop_duplicates(inplace=True)
    confirmation_df.fillna(DBM_NA_FILL_VALUE, inplace=True)
    confirmation_df = get_scaled_df_from_scaler(confirmation_df, scaler)
    return confirmation_df


def show_classifiers_confusion_matrix_on_validation_set(X_train, y_train, X_test, y_test):
    for classifier_data in CLASSIFIERS_FOR_EVALUATION:
        classifier = classifier_data["classifier"]
        name = classifier_data["name"]
        classifier.fit(X_train, y_train)
        y_predictions = classifier.predict(X_test)
        # confusion matrix code from https://stackoverflow.com/a/50329263/7782700
        labels = classifier.classes_
        matrix_df = pd.DataFrame(confusion_matrix(y_test, y_predictions), columns=labels, index=labels)
        matrix_df = matrix_df.add_prefix("pred_")
        matrix_df.to_csv(os.path.join("confusion_matrices", name + ".csv"))
        # print(matrix_df)


if __name__ == "__main__":
    if SUPPRESS_ALL_WARNINGS:
        warnings.filterwarnings("ignore")
    master_df, min_max_scaler = get_scaled_data_and_scaler(
        load_and_clean_combined_data_df_from_file_name(COMBINED_DATA_EXPORT_FILE_NAME))
    train_df, validation_df = split_into_train_and_validation_df(master_df)
    validation_df = oversample_df_if_enabled(validation_df)
    X_train, y_train, X_validation, y_validation = get_X_and_y_for_train_and_validation(train_df, validation_df)
    assert_train_and_test_feature_vectors_are_fully_distinct(X_train, X_validation)
    save_feature_array_column_order(X_train)
    save_model_and_scaler(fit_model_on_test_and_validation_df(MODEL_TO_SERIALIZE, train_df, validation_df),
                          min_max_scaler)
    conduct_random_search_if_enabled(X_train, y_train)
    if SHOW_VALIDATION_SET_PERFORMANCE:
        report_classifiers_performance_on_validation_set(X_train, y_train, X_validation, y_validation)
    if SHOW_KFOLD_CROSS_VALIDATION_PERFORMANCE:
        report_classifiers_performance_on_stratified_kfold(master_df)
    # test performance on new data
    confirmation_df = load_confirmation_df_into_feature_vectors_using_scaler_based_on_master_df_classes(min_max_scaler,
                                                                                                        master_df[
                                                                                                            "nodeId"].unique())
    if SHOW_CONFIRMATION_SET_PERFORMANCE:
        X_confirmation, y_confirmation = get_X_and_y_for_df(confirmation_df)
        X_master, y_master = get_X_and_y_for_df(master_df)
        print("info for confirmation dataset, using all master data for training")
        report_classifiers_performance_on_validation_set(X_master, y_master, X_confirmation, y_confirmation)
        show_classifiers_confusion_matrix_on_validation_set(X_master, y_master, X_confirmation, y_confirmation)
    pass

"""
K Nearest Neighbor
accuracy 0.9337349397590361
balanced accuracy 0.9163194444444444

Feed Forward Neural Network
accuracy 0.9337349397590361
balanced accuracy 0.9147569444444443

Multinomial Naive Bayes
accuracy 0.9036144578313253
balanced accuracy 0.8908854166666667

Random Forest
accuracy 0.9759036144578314
balanced accuracy 0.96875

SVC
accuracy 0.9397590361445783
balanced accuracy 0.9225694444444443

AdaBoost Classifier
accuracy 0.9518072289156626
balanced accuracy 0.9434027777777778

Gradient Boosting Classifier
accuracy 0.9518072289156626
balanced accuracy 0.9432291666666667

Bagging Classifier
accuracy 0.9457831325301205
balanced accuracy 0.9354166666666667

"""
