from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

EXPORT_FILE_NAME = "combined_data.csv"

master_df = pd.read_csv(EXPORT_FILE_NAME)
print("min dBm: " + str(min(master_df.min(axis=1))))
master_df.fillna(-100, inplace=True) #-85 was the best so far? use -100 though
classification_target = master_df['nodeId']
master_df.drop(['Unnamed: 0', 'timestamp', 'nodeId'], axis=1, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
master_df = pd.DataFrame(min_max_scaler.fit_transform(master_df), columns=master_df.columns, index=master_df.index) #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

X_train, X_test, y_train, y_test = train_test_split(master_df, classification_target, test_size=0.33, stratify=classification_target, random_state=36) #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

classifiers = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": KNeighborsClassifier(n_neighbors=3)
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500, random_state=42)
    },
    {
        "name": "Multinomial Naive Bayes",
        "classifier": MultinomialNB()
    }
]

for classifier in classifiers:
    classifier["classifier"].fit(X_train, y_train)
    y_pred = classifier["classifier"].predict(X_test)
    print(classifier["name"] + ": " + str(accuracy_score(y_test, y_pred)))

#print(len(y_train))

print("doing 5-fold cross validation now")
master_df = master_df.to_numpy()
classification_target = classification_target.to_numpy()
kf = StratifiedKFold(n_splits=4)
classifiers = [
    {
        "name": "K(3) Nearest Neighbor",
        "classifier": KNeighborsClassifier(n_neighbors=3)
    },
    {
        "name": "Feed Forward Neural Network",
        "classifier": MLPClassifier(hidden_layer_sizes=140, max_iter=500, random_state=42)
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
        classifier.fit(master_df[train_index], classification_target[train_index])
        y_pred = classifier.predict(master_df[test_index])
        accuracies.append(accuracy_score(y_pred, classification_target[test_index]))
        #print(len(train_index))
    print(classifier_package["name"] + ": " + str(accuracies))


#output:
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