import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler 

#Load csv files, flatten y to 1D array
x = pd.read_csv('processed_x.csv')
y = pd.read_csv('processed_y.csv').values.ravel()


#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)  # Scaling all data at once

#fit model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

#predict
y_pred = model.predict(X_scaled)
y_probs = model.predict_proba(X_scaled)[:, 1] 

#get accuracy 
accuracy = accuracy_score(y, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
auc_score = roc_auc_score(y, y_probs)
print(f"AUC-ROC score: {auc_score:.4f}")