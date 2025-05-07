import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("The first Pokemon makes the first move.\n")
#inputs
p1name = input("What is the name of the first Pokemon?")
p1type1 = input("What is the first Pokemon's primary type? The type can be Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, or Fairy.")
p1type2 = input("What is the first Pokemon's secondary type? The type can be Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, or Fairy.")
p1hp = input("What is the first Pokemon's HP?")
p1atk = input("What is the first Pokemon's Attack?")
p1def = input("What is the first Pokemon's Defense?")
p1spatk = input("What is the first Pokemon's Special Attack?")
p1spdef = input("What is the first Pokemon's Special Defense?")
p1speed = input("What is the first Pokemon's Speed?")
p1gen = input("What generation number is this Pokemon from?")
p1leg = input("Is the first Pokemon legendary? Y/N")

p2name = input("What is the name of the second Pokemon?")
p2type1 = input("What is the second Pokemon's primary type? The type can be Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, or Fairy.")
p2type2 = input("What is the second Pokemon's secondary type? The type can be Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, or Fairy.")
p2hp = input("What is the second Pokemon's HP?")
p2atk = input("What is the second Pokemon's Attack?")
p2def = input("What is the second Pokemon's Defense?")
p2spatk = input("What is the second Pokemon's Special Attack?")
p2spdef = input("What is the second Pokemon's Special Defense?")
p2speed = input("What is the second Pokemon's Speed?")
p2gen = input("What generation number is this Pokemon from?")
p2leg = input("Is the second Pokemon legendary? Y/N")


#create dictionaries for both pokemon
pokemon1 = {
    'HP': int(p1hp),
    'Attack': int(p1atk),
    'Defense': int(p1def),
    'Sp. Atk': int(p1spatk),
    'Sp. Def': int(p1spdef),
    'Speed': int(p1speed),
    'Generation': int(p1gen),
    'Legendary': 1 if p1leg.upper() == 'Y' else 0
}

# Create dictionary for second Pokemon
pokemon2 = {
    'HP': int(p2hp),
    'Attack': int(p2atk),
    'Defense': int(p2def),
    'Sp. Atk': int(p2spatk),
    'Sp. Def': int(p2spdef),
    'Speed': int(p2speed),
    'Generation': int(p2gen),
    'Legendary': 1 if p2leg.upper() == 'Y' else 0
}

#one hot encode types
all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 
             'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 
             'Steel', 'Fairy']
for type_name in all_types:
    pokemon1[f'type1_{type_name}'] = 1 if p1type1 == type_name else 0
    pokemon1[f'type2_{type_name}'] = 1 if p1type2 == type_name else 0
    pokemon2[f'type1_{type_name}'] = 1 if p2type1 == type_name else 0
    pokemon2[f'type2_{type_name}'] = 1 if p2type2 == type_name else 0

#make dataframe
battle_df = pd.DataFrame()
for key in pokemon1:
    battle_df[key] = [pokemon1[key]]
for key in pokemon2:
    battle_df[f'{key}_second'] = [pokemon2[key]]

scaler = joblib.load('scaler.joblib')
model = joblib.load('best_mlp_model.joblib')

X_scaled = scaler.transform(battle_df)

# predict
prediction = model.predict(X_scaled)
winner = p1name if prediction[0] == 1 else p2name
print(f"\nThe Pokemachine predicts that {winner} wins!")