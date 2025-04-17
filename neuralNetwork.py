import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
X = pd.read_csv("processed_x.csv")
y = pd.read_csv("processed_y.csv").values.ravel()

# Drop unnamed index columns if they exist
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

# OPTIONAL: Scale features for better MLP performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# OPTIONAL: Split data for evaluation (since MLP has no built-in evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model
model = MLPClassifier(hidden_layer_sizes=(128, 64),
                      activation='relu',
                      solver='adam',
                      alpha=0.001,
                      learning_rate_init=0.001,
                      max_iter=200,
                      random_state=42,
                      verbose=True)

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For AUC-ROC

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# AUC-ROC (only valid for binary classification)
if len(set(y)) == 2:
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC Score: {auc:.4f}")
else:
    print("AUC-ROC not available: not a binary classification problem.")
