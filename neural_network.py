import pandas as pd
from time import time
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

col_names = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
             'att14', 'att15', 'att16', 'label']

DATA = pd.read_csv("final_data.csv", header=None, names=col_names)
feature_cols = col_names[:-1]

X = DATA[feature_cols]
y = DATA.label

accuracy = 0
f1 = 0
precision = 0
recall = 0

time_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(8, 4), max_iter=100, random_state=0)

clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

time = time() - time_start

accuracy = accuracy + accuracy_score(y_test, y_pred)
f1 = f1 + f1_score(y_test, y_pred, average='weighted')
precision = precision + precision_score(y_test, y_pred, average='weighted')
recall = recall + recall_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Execution Time :", time, "seconds")
print("Precision:", precision)
print("F1 Score:", f1)
print("Recall Score:", recall)
