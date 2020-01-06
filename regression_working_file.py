import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# From data file get only the following columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "traveltime", "goout", "health", "Walc"]]

# What are we trying to predict
predict = "G3"

# Exclude from from X predict variable
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split data and leave 10% for tests
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Apply linear regression
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

# Print result
print(acc)