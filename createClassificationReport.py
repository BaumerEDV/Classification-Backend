from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EXPORT_FILE_NAME = "combined_data.csv"

master_df = pd.read_csv(EXPORT_FILE_NAME)
print("min dBm: " + str(min(master_df.min(axis=1))))
master_df.fillna(-100, inplace=True) #-85 was the best so far?
classification_target = master_df['nodeId']
master_df.drop(['Unnamed: 0', 'timestamp', 'nodeId'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(master_df, classification_target, test_size=0.33, stratify=classification_target, random_state=42) #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

for classifier in [KNeighborsClassifier(n_neighbors=3), MLPClassifier(hidden_layer_sizes=140, max_iter=500, random_state=42)]:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))


"""
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
"""