import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score,  roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import RandomizedSearchCV

# Load data
X = pd.read_csv("processed_x.csv")
y = pd.read_csv("processed_y.csv").values.ravel()

# Drop unnamed index columns if they exist
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

# Scale features for better MLP performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split data for evaluation (since MLP has no built-in evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Define parameter space
param_distributions = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'max_iter': [100, 200, 300],
    'learning_rate': ['constant', 'adaptive']
}

# Define base model
base_model = MLPClassifier(learning_rate_init=0.001,
                      random_state=42,
                      verbose=True)


# Do random search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy',
    verbose=2
)

print("Starting randomized search CV\n")
random_search.fit(X_train, y_train)

print("Best params: ")
print(random_search.best_params_)

# Evaluate on test data
best_model = random_search.best_estimator_
test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_auc = roc_auc_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")


#save this model using joblib
joblib.dump(best_model, 'best_mlp.joblib2')


#Show seaborn ROC curve
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, sillybirt = roc_curve(y_test, y_pred_proba)

plt.figure()
sns.set_style('darkgrid')
sns.lineplot(x=fpr, y=tpr, color='red', label='ROC curve')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.savefig('roc_curve2.png')
plt.show()
