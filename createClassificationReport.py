from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    SCALER_JOBLIB_FILE_NAME, DBM_NA_FILL_VALUE
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier

K_FOLD_NUMBER = 4
RANDOMIZED_SEARCH_ITERATIONS = 10
SKIP_SEARCH = True
OVERSAMPLE_VALIDATION_DATASET = True
OVERSAMPLE_KFOLD_VALIDATION_DATASETS = True
SCALERS = [
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
    preprocessing.RobustScaler,
    preprocessing.MaxAbsScaler
]
SCALER = SCALERS[0]
OVERSAMPLE_ALL_CLASSIFIERS = False
SHOW_VALIDATION_SET_PERFORMANCE = False
SHOW_KFOLD_CROSS_VALIDATION_PERFORMANCE = True

CLASSIFIERS_WITH_HYPERPARAMETER_DISTRIBUTIONS = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", KNeighborsClassifier())
        ]),
        "param_dist": {
            "classification__n_neighbors": [3, 4, 5, 6],
            "classification__weights": ["uniform", "distance"],
            "classification__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "classification__leaf_size": [10, 20, 30, 75, 150, 300],
            "classification__p": [1, 2],
            "classification__n_jobs": [-1]
        }
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", MLPClassifier())
        ]),
        "param_dist": {
            "classification__hidden_layer_sizes": [50, 75, 100, 125, 140, 300, 450],
            "classification__activation": ["identity", "logistic", "tanh", "relu"],
            "classification__solver": ["lbfgs", "sgd", "adam"],
            "classification__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "classification__learning_rate": ["constant", "invscaling", "adaptive"],
            "classification__learning_rate_init": [1e-4, 1e-3, 1e-2],
            "classification__max_iter": [500, 700, 1000],
            "classification__shuffle": [False, True],
            "classification__random_state": [42],
            "classification__momentum": stats.uniform(0, 1),
            "classification__nesterovs_momentum": [False, True],
            "classification__beta_1": [0.9999, 0.9, 0.8, 0.5, 0.1],
            "classification__beta_2": [0.9999, 0.999, 0.99, 0.9],
        }
    },
    {
        "name": "Multinomial Naive Bayes",
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", MultinomialNB())
        ]),
        "param_dist": {
            "classification__alpha": [0, 0.5, 0.9, 1],
            "classification__fit_prior": [False, True]
        }
    },
    {
        "name": "Random Forest Classifier",
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", RandomForestClassifier())
        ]),
        "param_dist": {
            "classification__n_estimators": [10, 100, 150, 200, 300],
            "classification__criterion": ["gini", "entropy"],
            "classification__max_features": ["sqrt", "log2", "auto"],
            "classification__n_jobs": [-1],
            "classification__random_state": [42],
            "classification__class_weight": [None, "balanced"]
        }
    }
]
CLASSIFIERS_FOR_EVALUATION = [
    {
        "name": "K Nearest Neighbor",  # worse with pipeline
        "classifier": KNeighborsClassifier(weights='distance', p=1, n_neighbors=4,
                                           n_jobs=-1, leaf_size=300, algorithm='auto'),
    },
    {
        "name": "Feed Forward Neural Network",  # worse with pipeline
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                                    random_state=42),
    },
    {
        "name": "Multinomial Naive Bayes",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", MultinomialNB())
        ]),
    },
    {
        "name": "Random searched MLP optimal",  # worse with pipeline
        "classifier": MLPClassifier(activation='tanh', alpha=1e-05, beta_1=0.1, beta_2=0.99, hidden_layer_sizes=450,
                                    learning_rate='invscaling', learning_rate_init=0.001, max_iter=1000,
                                    momentum=0.5009070908512535, nesterovs_momentum=True, random_state=42,
                                    shuffle=False, solver='lbfgs')
    },
    {
        "name": "Random searched Random Forest optimal",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='sqrt',
                                                      criterion='gini', class_weight=None))
        ])
    },
    {
        "name": "SVC",  # equal with pipeline
        "classifier": SVC(kernel="linear", C=0.95, probability=True, random_state=42)
    },
    # {  # commented out because this one takes forever and only goes up to about 70%
    #    "name": "Gaussian Process Classifier", # better with pipeline
    #    "classifier": GaussianProcessClassifier(1.5 * RBF(10.0), random_state=42)
    # },
    {
        "name": "AdaBoost Classifier",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
                                                  n_estimators=200, random_state=42))
        ])
    },
    # {  # Features are collinear so this is useless
    #    "name": "QuadraticDiscriminantAnalysis", # worse with pipeline
    #    "classifier": QuadraticDiscriminantAnalysis()
    # },
    {
        "name": "Gradient Boosting Classifier",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            (
                "classification",
                GradientBoostingClassifier(random_state=42, n_estimators=400, max_depth=5, subsample=0.5))
        ])
    },
    {
        "name": "Bagging Classifier",  # significantly better with pipeline - ties for best classifier
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", BaggingClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                 base_estimator=DecisionTreeClassifier(random_state=42, max_depth=15)))
        ])
    },  # maybe try a voting classifier?
]
MODEL_TO_SERIALIZE = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='sqrt',
                                            criterion='gini',
                                            class_weight=None)


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
        "roc_auc": roc_auc_score(y_validation, y_predictions_proba, multi_class="ovo", average="weighted"),
        "f1_score": f1_score(y_validation, y_predictions, average="weighted"),
        "precision": precision_score(y_validation, y_predictions, average="weighted", zero_division=1),
        "recall": recall_score(y_validation, y_predictions, average="weighted")
    }
    return performance


def conduct_random_search(X_train, y_train):
    for classifier_data in CLASSIFIERS_WITH_HYPERPARAMETER_DISTRIBUTIONS:
        model = classifier_data["classifier"]
        param_dist = classifier_data["param_dist"]
        random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                           n_iter=RANDOMIZED_SEARCH_ITERATIONS,
                                           cv=K_FOLD_NUMBER
                                           # , scoring=make_scorer(f1_score, average="weighted")
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


def load_and_clean_combined_data_df():
    df = pd.read_csv(COMBINED_DATA_EXPORT_FILE_NAME)
    df.fillna(DBM_NA_FILL_VALUE, inplace=True)
    df.drop(['timestamp', 'Unnamed: 0'], axis=1, inplace=True)
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


if __name__ == "__main__":
    master_df, min_max_scaler = get_scaled_data_and_scaler(load_and_clean_combined_data_df())
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
    pass

"""
4-fold cross validation vs oversampled test sets

K Nearest Neighbor
Mean accuracy 0.8995535714285714
Mean roc_auc 0.9862516534391534
Mean f1_score 0.8780869512378316
Mean precision 0.9468253968253968
Mean recall 0.8995535714285714

Feed Forward Neural Network
Mean accuracy 0.890625
Mean roc_auc 0.9944302721088435
Mean f1_score 0.8589023138776428
Mean precision 0.9454267178362573
Mean recall 0.890625

Multinomial Naive Bayes
Mean accuracy 0.8794642857142857
Mean roc_auc 0.9958545918367347
Mean f1_score 0.8527100711734515
Mean precision 0.9260377209595959
Mean recall 0.8794642857142857

Random searched MLP optimal
Mean accuracy 0.8950892857142857
Mean roc_auc 0.9967474489795918
Mean f1_score 0.8695118024283078
Mean precision 0.9480034722222223
Mean recall 0.8950892857142857

Random searched Random Forest optimal
Mean accuracy 0.9434523809523809
Mean roc_auc 0.9986111111111111
Mean f1_score 0.9357018849206349
Mean precision 0.9563368055555556
Mean recall 0.9434523809523809

SVC
Mean accuracy 0.8950892857142857
Mean roc_auc 0.9955144557823129
Mean f1_score 0.8634596055443096
Mean precision 0.9488989400584795
Mean recall 0.8950892857142857

AdaBoost Classifier
Mean accuracy 0.8392857142857143
Mean roc_auc 0.9407283399470899
Mean f1_score 0.80304765820802
Mean precision 0.902428300865801
Mean recall 0.8392857142857143

Gradient Boosting Classifier
Mean accuracy 0.9285714285714286
Mean roc_auc 0.9949192176870748
Mean f1_score 0.9127604166666666
Mean precision 0.9539930555555556
Mean recall 0.9285714285714286

Bagging Classifier
Mean accuracy 0.9434523809523809
Mean roc_auc 0.9974229969765684
Mean f1_score 0.9315900428153717
Mean precision 0.9654513888888889
Mean recall 0.9434523809523809
"""