#OVERVIEW: this one-hot encodes the pokemon types in the csv file 

import pandas as pd

#load dataset
df = pd.read_csv('pokemon.csv')

#strip column names
df.columns = df.columns.str.strip()

#print(df.columns)

#one-hot encode
type1_encoded = pd.get_dummies(df['Type 1'], prefix='type1')
type2_encoded = pd.get_dummies(df['Type 2'], prefix='type2')

#define all pokemon types
all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 
             'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']

#maintain consistency btwn type1 and type2
type1_encoded = type1_encoded.reindex(columns=[f'type1_{t}' for t in all_types], fill_value=0)
type2_encoded = type2_encoded.reindex(columns=[f'type2_{t}' for t in all_types], fill_value=0)

#combine onehot encoded to og dataframe
df = pd.concat([df, type1_encoded, type2_encoded], axis=1)

#drop original columns
df = df.drop(columns=['Type 1', 'Type 2'])

#save here
df.to_csv('pokemon_data_encoded.csv', index=False)
