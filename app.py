from typing import BinaryIO

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

dataset = pd.read_csv('swimming.csv')
print(dataset)
x = dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values
# print(x[0])
y = dataset.iloc[:, [6]].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=100)
classifier = KNeighborsClassifier()

ytrain = np.ravel(ytrain)
classifier.fit(xtrain, ytrain)

testdata = np.array([[1, 1, 1, 0, 1, 1]])
pred = classifier.predict(testdata)
print(pred)

import pickle

file = open('knnswim.pkl', 'wb')
pickle.dump(classifier, file)
file.close()
