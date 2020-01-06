"""
Linear Regression
"""
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# From data file get only the following columns
data = data[
    ["G1", "G2", "G3", "studytime", "failures", "absences", "age", "traveltime", "goout", "health", "Dalc", "Walc",
     "Medu", "Fedu", "famrel"]]
# What are we trying to predict
predict = "G3"

# Exclude from from X predict variable
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split data 90% training/10% test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Create linear regression model
linear = linear_model.LinearRegression()

# Load model
pickle_in = open("predictionmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Train model
linear.fit(x_train, y_train)

# Save trained model on file
with open("predictionmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

# Print accuracy result
accuracy = linear.score(x_test, y_test)
print(accuracy)

# Output predictions in console
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "traveltime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
