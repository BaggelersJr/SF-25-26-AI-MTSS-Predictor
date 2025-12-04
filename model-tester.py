from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

print(clf.predict(X_test))

user_data = []

for feature in load_breast_cancer().feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_data.append(value)
X_new = np.array(user_data).reshape(1, -1)
prediction = clf.predict(X_new)[0]
print(f"Predicted class for the input data: {prediction}")
