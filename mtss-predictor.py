from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import shap 

df = pd.read_csv("synthetic_MTSS_dataset.csv")
X = df.drop("Injury", axis=1)
y = df["Injury"]
cv_accuracies = []
test_accuracies = []
for trial in range(500):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=10000))
    ])
    kfs = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kfs, scoring='accuracy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_accuracies.append(np.mean(scores))
    test_accuracies.append(accuracy)
cv_df = pd.DataFrame({"cv_accuracy": cv_accuracies})
test_df = pd.DataFrame({"test_accuracy": test_accuracies})
cv_df.to_csv("logistic_regression_cv_accuracies.csv", index=False)
test_df.to_csv("logistic_regression_test_accuracies.csv", index=False)
print(f"Average CV Accuracy over 500 trials: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print(f"Average Test Accuracy over 500 trials: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")

#Xtrains = model.named_steps['scaler'].transform(X_train)
#Xtests = model.named_steps['scaler'].transform(X_test)
#explainer = shap.LinearExplainer(model.named_steps['classifier'], Xtrains)
#shap_values = explainer.shap_values(Xtests)
#shap.summary_plot(shap_values, Xtests, feature_names=X.columns)
