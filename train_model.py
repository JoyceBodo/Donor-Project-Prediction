import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load dataset
data = pd.read_csv("donor_funded_projects.csv")  

# Data Preprocessing
encoder = LabelEncoder()
for col in ['funding_source', 'stakeholder_involvement']:
    data[col] = encoder.fit_transform(data[col])

features = ['funding_amount', 'duration', 'stakeholder_involvement', 'funding_source']
target = 'success'  # 1: Success, 0: Failure

X = data[features]
y = data[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save the best model
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save LabelEncoder & Scaler
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Best model saved as model.pkl")