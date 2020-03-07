from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import \
    roc_auc_score  # https://elitedatascience.com/imbalanced-classes
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
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

EXPORT_FILE_NAME = "combined_data.csv"
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"
K_FOLD_NUMBER = 4
RANDOMIZED_SEARCH_ITERATIONS = 30
SKIP_SEARCH = True


# from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
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
# print(master_df.groupby(master_df.columns.tolist(),as_index=False).size())
min_max_scaler = preprocessing.MinMaxScaler()
master_df[
    master_df.columns.difference(['nodeId'])] = min_max_scaler.fit_transform(
    master_df[master_df.columns.difference(['nodeId'])])
train_df, test_df = train_test_split(master_df, test_size=0.33, random_state=36,
                                     stratify=master_df['nodeId'])
print(pd.merge(train_df, test_df, on=train_df.columns.to_list(), how='outer',
               indicator='Exist')['Exist'].value_counts())
master_df = train_df
# upsampling
"""
most_common_class = master_df['nodeId'].value_counts().idxmax()
n_most_common_class = master_df['nodeId'].value_counts().max()
base_df = master_df.loc[master_df['nodeId'] == most_common_class]
new_df = pd.DataFrame()
for class_name in master_df['nodeId'].unique():
    if class_name == most_common_class:
        new_df = pd.concat([new_df, base_df], axis=0)
        continue
    n_class = (master_df.nodeId == class_name).sum()
    class_oversampled_df = master_df.loc[
        master_df['nodeId'] == class_name].sample(n_most_common_class,
                                                  replace=True, random_state=42)
    new_df = pd.concat([new_df, class_oversampled_df], axis=0)
    # https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
master_df = new_df
del new_df
del base_df
# print(master_df['nodeId'].unique())
"""
# end upsampling

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

X_train = master_df.drop(['nodeId'], axis=1)
y_train = master_df['nodeId']
X_test = test_df.drop(['nodeId'], axis=1)
y_test = test_df['nodeId']

# test if any training data is also in the test data
print(pd.merge(X_train, X_test, on=X_train.columns.to_list(), how='outer',
               indicator='Exist')['Exist'].value_counts())

save_feature_arry_column_order(X_train)

print("doing " + str(K_FOLD_NUMBER) + "-fold cross validation now")
classifiers = [
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

for classifier_data in classifiers:
    if SKIP_SEARCH:
        continue
    model = classifier_data["classifier"]
    param_dist = classifier_data["param_dist"]
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=RANDOMIZED_SEARCH_ITERATIONS,
                                       cv=K_FOLD_NUMBER
                                       # , scoring=make_scorer(f1_score, average="weighted")
                                       )
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)
"""
Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'distance', 'classification__p': 1, 'classification__n_neighbors': 4, 'classification__n_jobs': -1, 'classification__leaf_size': 150, 'classification__algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'uniform', 'classification__p': 1, 'classification__n_neighbors': 4, 'classification__n_jobs': -1, 'classification__leaf_size': 150, 'classification__algorithm': 'ball_tree'}

Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'distance', 'classification__p': 1, 'classification__n_neighbors': 4, 'classification__n_jobs': -1, 'classification__leaf_size': 30, 'classification__algorithm': 'kd_tree'}

Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'uniform', 'classification__p': 1, 'classification__n_neighbors': 4, 'classification__n_jobs': -1, 'classification__leaf_size': 300, 'classification__algorithm': 'brute'}

Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'uniform', 'classification__p': 1, 'classification__n_neighbors': 3, 'classification__n_jobs': -1, 'classification__leaf_size': 30, 'classification__algorithm': 'ball_tree'}

Model with rank: 1
Mean validation score: 0.890 (std: 0.043)
Parameters: {'classification__weights': 'uniform', 'classification__p': 1, 'classification__n_neighbors': 3, 'classification__n_jobs': -1, 'classification__leaf_size': 75, 'classification__algorithm': 'brute'}
---------------------
---------------------
sucks
Model with rank: 1
Mean validation score: 0.950 (std: 0.043)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.8004870606914816, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

good
Model with rank: 2
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.1, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.33678505066777575, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

optimal
Model with rank: 2
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.504547384522401, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}
---------------------------------
---------------------------------
Model with rank: 1
Mean validation score: 0.869 (std: 0.043)
Parameters: {'classification__fit_prior': False, 'classification__alpha': 0.5}

Model with rank: 1
Mean validation score: 0.869 (std: 0.043)
Parameters: {'classification__fit_prior': True, 'classification__alpha': 0.5}

Model with rank: 3
Mean validation score: 0.839 (std: 0.062)
Parameters: {'classification__fit_prior': False, 'classification__alpha': 0.9}

Model with rank: 3
Mean validation score: 0.839 (std: 0.062)
Parameters: {'classification__fit_prior': True, 'classification__alpha': 0.9}
----------------------------------
----------------------------------
best model so far
Model with rank: 1
Mean validation score: 0.940 (std: 0.045)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 200, 'classification__max_features': 'sqrt', 'classification__criterion': 'gini', 'classification__class_weight': None}

Model with rank: 1
Mean validation score: 0.940 (std: 0.020)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 200, 'classification__max_features': 'log2', 'classification__criterion': 'gini', 'classification__class_weight': 'balanced'}

Model with rank: 1
Mean validation score: 0.940 (std: 0.020)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 100, 'classification__max_features': 'log2', 'classification__criterion': 'gini', 'classification__class_weight': None}

Model with rank: 1
Mean validation score: 0.940 (std: 0.045)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 200, 'classification__max_features': 'sqrt', 'classification__criterion': 'gini', 'classification__class_weight': 'balanced'}

Model with rank: 1
Mean validation score: 0.940 (std: 0.020)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 150, 'classification__max_features': 'log2', 'classification__criterion': 'gini', 'classification__class_weight': 'balanced'}

Model with rank: 1
Mean validation score: 0.940 (std: 0.020)
Parameters: {'classification__random_state': 42, 'classification__n_jobs': -1, 'classification__n_estimators': 150, 'classification__max_features': 'sqrt', 'classification__criterion': 'gini', 'classification__class_weight': 'balanced'}

"""

"""
After 1000 random runs:
doesnt work
Model with rank: 1
Mean validation score: 0.950 (std: 0.033)
Parameters: {'classification__activation': 'relu', 'classification__alpha': 1e-05, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.024662022810686413, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'sgd'}

doesnt work
Model with rank: 2
Mean validation score: 0.940 (std: 0.034)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.01, 'classification__beta_1': 0.1, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.8186512105157959, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

works but isn't better than what I generated already
Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.4760240076094405, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

works but no improvement
Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.25894866798476657, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

works but doesnt improve
Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.22458085657377913, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 1e-05, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 75, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.8094376719579176, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 700, 'classification__momentum': 0.25339349128370736, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 75, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.9650340871315923, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.5, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.11530479402625593, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.06531245069723013, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 1e-05, 'classification__beta_1': 0.1, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 700, 'classification__momentum': 0.09881828774312418, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.7430288508568218, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.34098448591511876, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 700, 'classification__momentum': 0.9062746678823889, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.1656816404762106, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 50, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.9230044541445139, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.929721361338965, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.146149341171938, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'relu', 'classification__alpha': 0.01, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 500, 'classification__momentum': 0.9433021066364697, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'relu', 'classification__alpha': 0.01, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.6503265055352974, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.10580799159557763, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.9945415819321388, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.9, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.9118362481376419, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.0001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.5371537712029393, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.633146972990548, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 500, 'classification__momentum': 0.7346143898591809, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.3385191627388955, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.01, 'classification__beta_1': 0.1, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 50, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.2592049527751651, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 500, 'classification__momentum': 0.2807290186745015, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.6203996284111076, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.9, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 50, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.8402733869469622, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.07108381812926767, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.34672334261017734, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.9867900387576288, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.0001, 'classification__beta_1': 0.1, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.00959300244343253, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.0001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.39253698578813734, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.0001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.5396779136245641, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.211777714429764, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 50, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.2862021850478337, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.1, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 700, 'classification__momentum': 0.032102527721778906, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.01, 'classification__beta_1': 0.5, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 100, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.834882710338749, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 50, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.4237359589313181, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.04095485436573909, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.6042088422819918, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 700, 'classification__momentum': 0.5530417508074049, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.001, 'classification__beta_1': 0.9, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 500, 'classification__momentum': 0.628132046254909, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.1, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.6943704235999095, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 0.0001, 'classification__beta_1': 0.5, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 700, 'classification__momentum': 0.5344845169258648, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 0.001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 1000, 'classification__momentum': 0.8234148558577793, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.035)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.5, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'invscaling', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.07552092001067101, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'relu', 'classification__alpha': 0.001, 'classification__beta_1': 0.8, 'classification__beta_2': 0.9, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 500, 'classification__momentum': 0.5587156430226585, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'adam'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'tanh', 'classification__alpha': 1e-05, 'classification__beta_1': 0.1, 'classification__beta_2': 0.9999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.919908204787091, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.045)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.1, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 125, 'classification__learning_rate': 'constant', 'classification__learning_rate_init': 0.001, 'classification__max_iter': 700, 'classification__momentum': 0.013431232732374232, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'logistic', 'classification__alpha': 0.01, 'classification__beta_1': 0.8, 'classification__beta_2': 0.999, 'classification__hidden_layer_sizes': 140, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.01, 'classification__max_iter': 1000, 'classification__momentum': 0.9384790514450873, 'classification__nesterovs_momentum': False, 'classification__random_state': 42, 'classification__shuffle': True, 'classification__solver': 'sgd'}

Model with rank: 3
Mean validation score: 0.939 (std: 0.021)
Parameters: {'classification__activation': 'identity', 'classification__alpha': 1e-05, 'classification__beta_1': 0.9, 'classification__beta_2': 0.99, 'classification__hidden_layer_sizes': 450, 'classification__learning_rate': 'adaptive', 'classification__learning_rate_init': 0.0001, 'classification__max_iter': 1000, 'classification__momentum': 0.5153069227852162, 'classification__nesterovs_momentum': True, 'classification__random_state': 42, 'classification__shuffle': False, 'classification__solver': 'lbfgs'}


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

classifier = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_features='sqrt', criterion='gini', class_weight=None)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_predictions, multi_class="ovo",
                    average="weighted"))  # https://elitedatascience.com/imbalanced-classes
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
print("----")

#current best:
classifier = MLPClassifier(activation='tanh', alpha=1e-05, beta_1=0.1, beta_2=0.99, hidden_layer_sizes=450,
                           learning_rate='invscaling', learning_rate_init=0.001, max_iter=1000,
                           momentum=0.5009070908512535, nesterovs_momentum=True, random_state=42, shuffle=False,
                           solver='lbfgs')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_predictions, multi_class="ovo",
                    average="weighted"))  # https://elitedatascience.com/imbalanced-classes
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
print("----")

classifier = KNeighborsClassifier(weights='distance', p=1, n_neighbors=4,
                                  n_jobs=-1, leaf_size=300, algorithm='auto')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(accuracy_score(y_test, y_predictions))
y_predictions = classifier.predict_proba(X_test)
print(
    roc_auc_score(y_test, y_predictions, multi_class="ovo", average="weighted"))
print(f1_score(y_test, classifier.predict(X_test), average="weighted"))

classifiers = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": KNeighborsClassifier(n_neighbors=3),
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                                    random_state=42),
    },
    {
        "name": "Multinomial Naive Bayes",
        "classifier": MultinomialNB(),
    }
]
for classifier_data in classifiers:
    classifier = classifier_data["classifier"]
    print(classifier_data["name"])
    classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_test)
    print(accuracy_score(y_test, y_predictions))
    y_predictions = classifier.predict_proba(X_test)
    print(roc_auc_score(y_test, y_predictions, multi_class="ovo",
                        average="weighted"))
    print(f1_score(y_test, classifier.predict(X_test), average="weighted"))
