import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib  #for model saving just in case

#load csv
X = pd.read_csv("processed_x.csv")
y = pd.read_csv("processed_y.csv")

#convert y
y = y.values.ravel()

#drop unnamed index cols if exist
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

#train on entire dataset
model = MLPClassifier(hidden_layer_sizes=(128, 64),
                      activation='relu',
                      solver='adam',
                      alpha=0.001,
                      learning_rate_init=0.001,
                      max_iter=200,
                      random_state=42,
                      verbose=True)

model.fit(X, y)

#save model- this is optional but i was like ok maybe in case we want to access it again
#joblib.dump(model, 'pokemon_battle_model.pkl')

print("Model trained on all data")
