import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

#Load csv files
df_battles = pd.read_csv('combats.csv')
df_pokemon = pd.read_csv('pokemon_data_encoded.csv')

#Set index for merging
df_pokemon.set_index('#', inplace=True)

#Merge stats for both Pokémon with combat outcomes
df = df_battles.copy()
df = df.merge(df_pokemon, left_on='First_pokemon', right_index=True, suffixes=('_first', ''))
df = df.merge(df_pokemon, left_on='Second_pokemon', right_index=True, suffixes=('', '_second'))

#1 if First pokemon wins, 0 otherwise
df['First_wins'] = (df['Winner'] == df['First_pokemon']).astype(int)

#drop id columns
columns_to_drop = ['First_pokemon', 'Second_pokemon', 'Winner']
columns_to_drop += [col for col in df.columns if 'Name' in col]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#split feats/target
X = df.drop('First_wins', axis=1)
y = df['First_wins']

#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling all data at once

#fit model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

#predict
y_pred = model.predict(X_scaled)

#get accuracy 
accuracy = accuracy_score(y, y_pred)

print(f"Model accuracy: {accuracy:.4f}")
