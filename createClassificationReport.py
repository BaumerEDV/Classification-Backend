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

EXPORT_FILE_NAME = "combined_data.csv"
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"
K_FOLD_NUMBER = 4


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
        "classifier": KNeighborsClassifier(n_neighbors=3)
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500,
                                    random_state=42)
    },
    {
        "name": "Multinomial Naive Bayes",
        "classifier": MultinomialNB()
    }
]
for classifier_package in classifiers:
    accuracies = []
    classifier = classifier_package["classifier"]
    for train_index, test_index in kf.split(master_df, classification_target):
        classifier.fit(master_df[train_index],
                       classification_target[train_index])
        y_predictions = classifier.predict(master_df[test_index])
        accuracies.append(
            accuracy_score(y_predictions, classification_target[test_index]))
        # print(len(train_index))
    print(classifier_package["name"] + ": " + str(median(accuracies)))

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
