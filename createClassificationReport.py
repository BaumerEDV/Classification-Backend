from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from statistics import median, mean, stdev
from joblib import dump
import numpy as np
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


EXPORT_FILE_NAME = "combined_data.csv"
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"
K_FOLD_NUMBER = 4
RANDOMIZED_SEARCH_ITERATIONS = 100


#from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def save_feature_arry_column_order(feature_array):
    feature_array[[False] * len(feature_array)].to_csv(
        EXPORT_FEATURE_VECTOR_FILE_NAME, index=False)


master_df = pd.read_csv(EXPORT_FILE_NAME)
master_df.fillna(-100, inplace=True)
classification_target = master_df['nodeId']
master_df.drop(['Unnamed: 0', 'timestamp', 'nodeId'], axis=1, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
master_df = pd.DataFrame(min_max_scaler.fit_transform(master_df),
                         columns=master_df.columns,
                         index=master_df.index)  # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

X_train, X_test, y_train, y_test = train_test_split(master_df,
                                                    classification_target,
                                                    test_size=0.33,
                                                    stratify=classification_target,
                                                    random_state=36)  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

save_feature_arry_column_order(X_train)

print("doing " + str(K_FOLD_NUMBER) + "-fold cross validation now")
master_df = master_df.to_numpy()
classification_target = classification_target.to_numpy()
kf = StratifiedKFold(n_splits=K_FOLD_NUMBER)
classifiers = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": KNeighborsClassifier(),
        "param_dist": {
            "n_neighbors": [3, 4, 5, 6],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 75, 150, 300],
            "p": [1, 2],
            "n_jobs": [-1]
        }
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": MLPClassifier(),
        "param_dist": {
            "hidden_layer_sizes": [50, 75, 100, 125, 140, 300, 450],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "max_iter": [100, 200, 500, 1000],
            "shuffle": [False, True],
            "random_state": [42],
            "momentum": stats.uniform(0, 1),
            "nesterovs_momentum": [False, True],
            "beta_1": [1, 0.9, 0.8, 0.5, 0.1],
            "beta_2": [0.9999, 0.999, 0.99, 0.9],
        }
    },
    {
        "name": "Multinomial Naive Bayes",
        "classifier": MultinomialNB(),
        "param_dist": {
            "alpha": [0, 0.5, 0.9, 1],
            "fit_prior": [False, True]
        }
    }
]
for classifier_package in classifiers:
    accuracies = []
    classifier = classifier_package["classifier"]
    param_dist = classifier_package["param_dist"]
    random_search = RandomizedSearchCV(classifier, param_distributions=param_dist,
                                       n_iter=RANDOMIZED_SEARCH_ITERATIONS,
                                       cv=K_FOLD_NUMBER)
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)

#output:
"""
Model with rank: 1
Mean validation score: 0.905 (std: 0.033)
Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 150, 'algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.905 (std: 0.033)
Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 30, 'algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.905 (std: 0.033)
Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 10, 'algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.905 (std: 0.033)
Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 10, 'algorithm': 'kd_tree'}

Model with rank: 1
Mean validation score: 0.905 (std: 0.033)
Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 10, 'algorithm': 'ball_tree'}

----

Model with rank: 1
Mean validation score: 0.953 (std: 0.049)
Parameters: {'activation': 'logistic', 'alpha': 0.001, 'beta_1': 0.9, 'beta_2': 0.99, 'hidden_layer_sizes': 125, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 200, 'momentum': 0.3708563089326743, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.953 (std: 0.031)
Parameters: {'activation': 'logistic', 'alpha': 0.01, 'beta_1': 0.8, 'beta_2': 0.9, 'hidden_layer_sizes': 75, 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 200, 'momentum': 0.731910308820929, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': True, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.953 (std: 0.031)
Parameters: {'activation': 'tanh', 'alpha': 0.0001, 'beta_1': 0.8, 'beta_2': 0.99, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'max_iter': 1000, 'momentum': 0.272082676321058, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.953 (std: 0.031)
Parameters: {'activation': 'logistic', 'alpha': 0.0001, 'beta_1': 0.8, 'beta_2': 0.9, 'hidden_layer_sizes': 140, 'learning_rate': 'invscaling', 'learning_rate_init': 0.001, 'max_iter': 200, 'momentum': 0.21675198639412896, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': True, 'solver': 'adam'}

----

Model with rank: 1
Mean validation score: 0.828 (std: 0.058)
Parameters: {'fit_prior': False, 'alpha': 0}

Model with rank: 1
Mean validation score: 0.828 (std: 0.058)
Parameters: {'fit_prior': True, 'alpha': 0}

Model with rank: 3
Mean validation score: 0.762 (std: 0.033)
Parameters: {'fit_prior': False, 'alpha': 0.5}

Model with rank: 3
Mean validation score: 0.762 (std: 0.033)
Parameters: {'fit_prior': True, 'alpha': 0.5}


"""

# output:
"""""
min dBm: -97.0
K(3) Nearest Neighbor: 0.9433962264150944
Feed Forward Neural Network: 0.9056603773584906
Multinomial Naive Bayes: 0.7924528301886793
doing 5-fold cross validation now
K(3) Nearest Neighbor: [0.875, 0.975, 0.9230769230769231, 0.9487179487179487]
Feed Forward Neural Network: [0.95, 1.0, 0.9487179487179487, 0.8461538461538461]
Multinomial Naive Bayes: [0.8, 0.825, 0.7948717948717948, 0.6666666666666666]

Process finished with exit code 0
"""""

"""""
classifier = MLPClassifier(hidden_layer_sizes=140, max_iter=500, random_state=42)
#classifier = KNeighborsClassifier(3)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
for i in range(len(probabilities)):
    probability_set = probabilities[i]
    #print(probability_set)
    print("Correct Answer: " + str(y_test.to_numpy()[i]))
    for k in range(len(classifier.classes_)):
        print(classifier.classes_[k] + ": " + str(probability_set[k]))
"""""

# at this point save the best classifier using all the training data; as well as the scaler
classifier = MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                           random_state=42)
# classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(master_df, classification_target)

dump(classifier, "classifier.joblib")
dump(min_max_scaler, "scaler.joblib")
