from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

df = pd.read_csv("synthetic_MTSS_dataset.csv")
X = df.drop("Injury", axis=1)
y = df["Injury"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")
