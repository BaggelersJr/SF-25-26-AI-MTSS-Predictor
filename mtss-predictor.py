from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

df = pd.read_csv("synthetic_MTSS_dataset.csv")
X = df.drop("Injury", axis=1)
y = df["Injury"]
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=10000, random_state=23, class_weight='balanced'))
])
kfs = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
scores = cross_val_score(model, X, y, cv=kfs, scoring='accuracy')
print(f"Cross-validated accuracy scores: {scores}")
print(f"Mean cross-validated accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")
 