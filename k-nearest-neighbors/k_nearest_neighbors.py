"""
K-Nearest-Neighbor
"""
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("car.data")

# Label string data with numerical values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["safety"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Split data 90% training/10% test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Create model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(x_train, y_train)

# Calculate accuracy
accuracy = model.score(x_test, y_test)
print(accuracy)

# Output predictions to console
predicted = model.predict(x_test)
names = ["unacc", "accuracy", "good", "vgood"]

for i in range(len(predicted)):
    print(f"Predicted: {names[predicted[i]]}, Data: {x_test[i]}, Actual: {names[y_test[i]]}")
