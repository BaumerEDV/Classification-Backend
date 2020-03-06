from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score #https://elitedatascience.com/imbalanced-classes
from sklearn.metrics import f1_score, make_scorer
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


EXPORT_FILE_NAME = "combined_data.csv"
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"
K_FOLD_NUMBER = 4
RANDOMIZED_SEARCH_ITERATIONS = 30


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
master_df.drop(['timestamp', 'Unnamed: 0'], axis=1, inplace=True)
master_df.drop_duplicates(inplace=True)
#print(master_df.groupby(master_df.columns.tolist(),as_index=False).size())
min_max_scaler = preprocessing.MinMaxScaler()
master_df[master_df.columns.difference(['nodeId'])] = min_max_scaler.fit_transform(master_df[master_df.columns.difference(['nodeId'])])
train_df, test_df = train_test_split(master_df, test_size=0.33, random_state=36, stratify=master_df['nodeId'])
print(pd.merge(train_df, test_df, on=train_df.columns.to_list(), how='outer', indicator='Exist')['Exist'].value_counts())
master_df = train_df
#upsampling
most_common_class = master_df['nodeId'].value_counts().idxmax()
n_most_common_class = master_df['nodeId'].value_counts().max()
base_df = master_df.loc[master_df['nodeId'] == most_common_class]
new_df = pd.DataFrame()
for class_name in master_df['nodeId'].unique():
    if class_name == most_common_class:
        new_df = pd.concat([new_df, base_df], axis=0)
        continue
    n_class = (master_df.nodeId == class_name).sum()
    class_oversampled_df = master_df.loc[master_df['nodeId'] == class_name].sample(n_most_common_class, replace=True, random_state=42)
    new_df = pd.concat([new_df, class_oversampled_df], axis=0)
    #https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
master_df = new_df
del new_df
del base_df
#print(master_df['nodeId'].unique())
#end upsampling

"""
#upsampling of test df
most_common_class = test_df['nodeId'].value_counts().idxmax()
n_most_common_class = test_df['nodeId'].value_counts().max()
base_df = test_df.loc[test_df['nodeId'] == most_common_class]
new_df = pd.DataFrame()
for class_name in test_df['nodeId'].unique():
    if class_name == most_common_class:
        new_df = pd.concat([new_df, base_df], axis=0)
        continue
    n_class = (test_df.nodeId == class_name).sum()
    class_oversampled_df = test_df.loc[test_df['nodeId'] == class_name].sample(n_most_common_class, replace=True, random_state=42)
    new_df = pd.concat([new_df, class_oversampled_df], axis=0)
    #https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
test_df = new_df
del new_df
del base_df
#print(master_df['nodeId'].unique())
#end upsampling
"""



#classification_target = master_df['nodeId']
#master_df.drop(['Unnamed: 0', 'timestamp', 'nodeId'], axis=1, inplace=True)
#min_max_scaler = preprocessing.MinMaxScaler()
#master_df = pd.DataFrame(min_max_scaler.fit_transform(master_df),
#                         columns=master_df.columns,
#                         index=master_df.index)  # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

#X_train, X_test, y_train, y_test = train_test_split(master_df,
#                                                    classification_target,
#                                                    test_size=0.33,
#                                                    stratify=classification_target,
#                                                    random_state=36)  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train = master_df.drop(['nodeId'], axis=1)
y_train = master_df['nodeId']
X_test = test_df.drop(['nodeId'], axis=1)
y_test = test_df['nodeId']

#test if any training data is also in the test data
#print(len(X_train))
#print(len(X_test))
#print(X_train.columns.to_list())
print(pd.merge(X_train, X_test, on=X_train.columns.to_list(), how='outer', indicator='Exist')['Exist'].value_counts())
#print(set(X_train.columns)==set(X_test.columns))

save_feature_arry_column_order(X_train)

print("doing " + str(K_FOLD_NUMBER) + "-fold cross validation now")
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
            "max_iter": [500, 700, 1000],
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
# TODO: oversampling leads to training data being in test data in the kfold cross validation
for classifier_package in classifiers:
    continue
    accuracies = []
    classifier = classifier_package["classifier"]
    param_dist = classifier_package["param_dist"]
    random_search = RandomizedSearchCV(classifier, param_distributions=param_dist,
                                       n_iter=RANDOMIZED_SEARCH_ITERATIONS,
                                       cv=K_FOLD_NUMBER
                                       #, scoring=make_scorer(f1_score, average="weighted")
                                       )
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)

#output:
"""
Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 3, 'n_jobs': -1, 'leaf_size': 150, 'algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 4, 'n_jobs': -1, 'leaf_size': 30, 'algorithm': 'ball_tree'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 4, 'n_jobs': -1, 'leaf_size': 20, 'algorithm': 'kd_tree'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 4, 'n_jobs': -1, 'leaf_size': 10, 'algorithm': 'kd_tree'}
------
Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'relu', 'alpha': 0.001, 'beta_1': 0.5, 'beta_2': 0.9999, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 500, 'momentum': 0.6980424280512796, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'relu', 'alpha': 0.0001, 'beta_1': 0.9, 'beta_2': 0.999, 'hidden_layer_sizes': 50, 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'max_iter': 1000, 'momentum': 0.1494587234719731, 'nesterovs_momentum': False, 'random_state': 42, 'shuffle': True, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'identity', 'alpha': 0.01, 'beta_1': 0.5, 'beta_2': 0.999, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 500, 'momentum': 0.7455215801287599, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': True, 'solver': 'adam'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'identity', 'alpha': 0.01, 'beta_1': 0.9, 'beta_2': 0.9999, 'hidden_layer_sizes': 75, 'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 500, 'momentum': 0.41418639699290927, 'nesterovs_momentum': False, 'random_state': 42, 'shuffle': True, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'tanh', 'alpha': 0.001, 'beta_1': 0.5, 'beta_2': 0.9999, 'hidden_layer_sizes': 100, 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 500, 'momentum': 0.16765559948854347, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'adam'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'logistic', 'alpha': 0.01, 'beta_1': 0.1, 'beta_2': 0.9, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.0001, 'max_iter': 700, 'momentum': 0.4913184988689062, 'nesterovs_momentum': False, 'random_state': 42, 'shuffle': False, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'relu', 'alpha': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'hidden_layer_sizes': 125, 'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 1000, 'momentum': 0.38656108682934, 'nesterovs_momentum': False, 'random_state': 42, 'shuffle': True, 'solver': 'adam'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'tanh', 'alpha': 0.0001, 'beta_1': 0.5, 'beta_2': 0.9999, 'hidden_layer_sizes': 450, 'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 1000, 'momentum': 0.5776675430336341, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'logistic', 'alpha': 0.01, 'beta_1': 0.8, 'beta_2': 0.99, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.0001, 'max_iter': 1000, 'momentum': 0.15692525314837436, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': False, 'solver': 'lbfgs'}

Model with rank: 1
Mean validation score: 0.996 (std: 0.006)
Parameters: {'activation': 'logistic', 'alpha': 0.01, 'beta_1': 0.1, 'beta_2': 0.9, 'hidden_layer_sizes': 140, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 700, 'momentum': 0.14873268357063663, 'nesterovs_momentum': True, 'random_state': 42, 'shuffle': True, 'solver': 'lbfgs'}
-----
Mean validation score: 0.979 (std: 0.016)
Parameters: {'fit_prior': False, 'alpha': 0}

Model with rank: 1
Mean validation score: 0.979 (std: 0.016)
Parameters: {'fit_prior': True, 'alpha': 0}

Model with rank: 3
Mean validation score: 0.963 (std: 0.032)
Parameters: {'fit_prior': False, 'alpha': 0.5}

Model with rank: 3
Mean validation score: 0.963 (std: 0.032)
Parameters: {'fit_prior': True, 'alpha': 0.5}



"""


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

master_df = pd.concat([master_df, test_df], axis=0)
classification_target = master_df['nodeId']
master_df = master_df.drop(['nodeId'], axis=1)

# at this point save the best classifier using all the training data; as well as the scaler
classifier = MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                           random_state=42)
# classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(master_df, classification_target)

dump(classifier, "classifier.joblib")
dump(min_max_scaler, "scaler.joblib")



classifier = MLPClassifier(activation='tanh', alpha=0.001, beta_1=0.5, beta_2=0.9999, hidden_layer_sizes=100, learning_rate='adaptive', learning_rate_init=0.01, max_iter=500, momentum=0.16765559948854347, nesterovs_momentum=True, random_state=42, shuffle=False, solver='adam')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_predictions, multi_class="ovo", average="weighted"))#https://elitedatascience.com/imbalanced-classes
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
print("----")

classifier = MLPClassifier(activation='relu', alpha=0.001, beta_1=0.8, beta_2=0.9999, hidden_layer_sizes=125, learning_rate='constant', learning_rate_init=0.0001, max_iter=1000, momentum=0.1662297780377231, nesterovs_momentum=True, random_state=42, shuffle=False, solver='lbfgs')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_predictions, multi_class="ovo", average="weighted"))#https://elitedatascience.com/imbalanced-classes
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
print("----")

classifier = KNeighborsClassifier(weights='distance', p=1, n_neighbors=4, n_jobs=-1, leaf_size=300, algorithm='auto')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_predictions, multi_class="ovo", average="weighted"))
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))

classifiers = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": KNeighborsClassifier(n_neighbors=3),
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
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                                    random_state=42),
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
for classifier_data in classifiers:
    classifier = classifier_data["classifier"]
    print(classifier_data["name"])
    classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_test)
    print(accuracy_score(y_test, y_predictions))
    y_predictions = classifier.predict_proba(X_test)
    print(roc_auc_score(y_test, y_predictions, multi_class="ovo", average="weighted"))
    print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
