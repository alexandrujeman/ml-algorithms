import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# From data file get only the following columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "traveltime", "goout", "health", "Dalc" ,"Walc", "Medu", "Fedu", "famrel"]]
# What are we trying to predict
predict = "G3"

# Exclude from from X predict variable
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split data and leave 10% for tests
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
# Create linear regression model
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

# Save/Load trained model on file
with open("predictionmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
"""
pickle_in = open("predictionmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Print accuracy result
accuracy = linear.score(x_test, y_test)
print(accuracy)

# Output predictions
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
