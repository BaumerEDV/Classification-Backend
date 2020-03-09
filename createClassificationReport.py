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

K_FOLD_NUMBER = 10
RANDOMIZED_SEARCH_ITERATIONS = 10
SKIP_SEARCH = True
OVERSAMPLE_VALIDATION_DATASET = True
OVERSAMPLE_KFOLD_VALIDATION_DATASETS = True
KFOLD_TEST_UPPER_STORIES_ONLY = True
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
    #{
    #    "name": "K Nearest Neighbor",  # worse with pipeline
    #    "classifier": KNeighborsClassifier(weights='distance', p=1, n_neighbors=4,
    #                                       n_jobs=-1, leaf_size=300, algorithm='auto'),
    #},
    #{
    #    "name": "Feed Forward Neural Network",  # worse with pipeline
    #    "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500,
    #                                random_state=42),
    #},
    #{
    #    "name": "Multinomial Naive Bayes",  # better with pipeline
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", MultinomialNB())
    #    ]),
    #},
    #{
    #    "name": "Random searched MLP",  # worse with pipeline
    #    "classifier": MLPClassifier(activation='tanh', alpha=1e-05, beta_1=0.1, beta_2=0.99, hidden_layer_sizes=450,
    #                                learning_rate='invscaling', learning_rate_init=0.001, max_iter=1000,
    #                                momentum=0.5009070908512535, nesterovs_momentum=True, random_state=42,
    #                                shuffle=False, solver='lbfgs')
    #},
    {
        "name": "Random searched Random Forest",  # better with pipeline
        "classifier": Pipeline([
            ("sampling", RandomOverSampler(random_state=42)),
            ("classification", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='sqrt',
                                                      criterion='gini', class_weight=None))
        ])
    },
    #{
    #    "name": "SVC",  # equal with pipeline
    #    "classifier": SVC(kernel="linear", C=0.95, probability=True, random_state=42)
    #},
    # {  # commented out because this one takes forever and only goes up to about 70%
    #    "name": "Gaussian Process Classifier", # better with pipeline
    #    "classifier": GaussianProcessClassifier(1.5 * RBF(10.0), random_state=42)
    # },
    #{
    #    "name": "AdaBoost Classifier",  # better with pipeline
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
    #                                              n_estimators=200, random_state=42))
    #    ])
    #},
    # {  # Features are collinear so this is useless
    #    "name": "QuadraticDiscriminantAnalysis", # worse with pipeline
    #    "classifier": QuadraticDiscriminantAnalysis()
    # },
    #{
    #    "name": "Gradient Boosting Classifier",  # better with pipeline
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        (
    #            "classification",
    #            GradientBoostingClassifier(random_state=42, n_estimators=400, max_depth=5, subsample=0.5))
    #    ])
    #},
    #{
    #    "name": "Bagging Classifier",  # significantly better with pipeline - ties for best classifier
    #    "classifier": Pipeline([
    #        ("sampling", RandomOverSampler(random_state=42)),
    #        ("classification", BaggingClassifier(n_estimators=200, random_state=42, n_jobs=-1,
    #                                             base_estimator=DecisionTreeClassifier(random_state=42, max_depth=15)))
    #    ])
    #},  # maybe try a voting classifier?
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
        "f1_score": f1_score(y_validation, y_predictions, average="weighted"),
        "precision": precision_score(y_validation, y_predictions, average="weighted", zero_division=1),
        "recall": recall_score(y_validation, y_predictions, average="weighted")
    }
    if not KFOLD_TEST_UPPER_STORIES_ONLY:
        performance["roc_auc"] = roc_auc_score(y_validation, y_predictions_proba, multi_class="ovo", average="weighted")
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
                filter = test_df['nodeId'].str.contains("((.*\.2\..*)|(.*\.3\..*))")
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
10-fold cross validation vs oversampled test sets

K Nearest Neighbor
Mean accuracy 0.9483193277310924
Mean roc_auc 0.9919332505729564
Mean f1_score 0.9415957444485946
Mean precision 0.9633945547916136
Mean recall 0.9483193277310924

Feed Forward Neural Network
Mean accuracy 0.9507352941176471
Mean roc_auc 0.9990252066735785
Mean f1_score 0.9453727420265847
Mean precision 0.9659520626432391
Mean recall 0.9507352941176471

Multinomial Naive Bayes
Mean accuracy 0.8835084033613445
Mean roc_auc 0.9982071805995124
Mean f1_score 0.8634469694893274
Mean precision 0.9177850213511978
Mean recall 0.8835084033613445

Random searched MLP optimal
Mean accuracy 0.9444327731092437
Mean roc_auc 0.9988593448743134
Mean f1_score 0.9372378745821652
Mean precision 0.9602630457777517
Mean recall 0.9444327731092437

Random searched Random Forest optimal
Mean accuracy 0.9688025210084034
Mean roc_auc 0.9997070276974426
Mean f1_score 0.9642325638858602
Mean precision 0.9789105502340797
Mean recall 0.9688025210084034

SVC
Mean accuracy 0.9588235294117647
Mean roc_auc 0.9994311360908
Mean f1_score 0.9535847723120078
Mean precision 0.968893818599701
Mean recall 0.9588235294117647

AdaBoost Classifier
Mean accuracy 0.9507878151260504
Mean roc_auc 0.9993759151387828
Mean f1_score 0.943926636664328
Mean precision 0.9687091666503431
Mean recall 0.9507878151260504

Gradient Boosting Classifier
Mean accuracy 0.9415966386554622
Mean roc_auc 0.996313496989705
Mean f1_score 0.9341724754539202
Mean precision 0.958873838579721
Mean recall 0.9415966386554622

Bagging Classifier
Mean accuracy 0.9434348739495798
Mean roc_auc 0.9991024250609335
Mean f1_score 0.9354250884539084
Mean precision 0.9597535389447154
Mean recall 0.9434348739495798
"""