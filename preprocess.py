'''
OVERVIEW:
Preprocesses the datasets to produce X and Y datasets that can be used for training and testing all of our models.

pokemon.csv and combats.csv from Weedle's Cave (https://www.kaggle.com/datasets/terminus7/pokemon-challenge)
must be in the same directory.

<(*)____//
 ( (___//
   ----
   ^ ^

note: also removes the pokedex number because homebrew pokemon wouldn't have one
'''

import pandas as pd

#load dataset
pokemon = pd.read_csv('pokemon.csv')
combats = pd.read_csv('combats.csv')

#strip column names of white space
pokemon.columns = pokemon.columns.str.strip()

#print(pokemon.columns)

#one-hot encode
type1_encoded = pd.get_dummies(pokemon['Type 1'], prefix='type1')
type2_encoded = pd.get_dummies(pokemon['Type 2'], prefix='type2')

#define all pokemon types
all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 
             'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']

#maintain consistency btwn type1 and type2
#make sure all types are present in type1 and type2
type1_encoded = type1_encoded.reindex(columns=[f'type1_{t}' for t in all_types], fill_value=0).astype(int)
type2_encoded = type2_encoded.reindex(columns=[f'type2_{t}' for t in all_types], fill_value=0).astype(int)

#combine onehot encoded to og dataframe
pokemon = pd.concat([pokemon, type1_encoded, type2_encoded], axis=1)

#drop original columns
pokemon = pokemon.drop(columns=['Type 1', 'Type 2'])

#merge w combat outcomes
pokemon.set_index('#', inplace=True)
combats = combats.merge(pokemon, left_on='First_pokemon', right_index = True, suffixes=('_first', ''))
combats = combats.merge(pokemon, left_on='Second_pokemon', right_index = True, suffixes=('', '_second'))

#1 if first pokemon wins, 0 otherwise
combats['First_wins'] = (combats['Winner'] == combats['First_pokemon']).astype(int)

#drop id columns
columns_to_drop = ['First_pokemon', 'Second_pokemon', 'Winner', '#']
columns_to_drop += [col for col in combats.columns if 'Name' in col]
combats.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#create x and y datasets
dfX = combats.drop('First_wins', axis=1)
dfY = combats['First_wins']

#save as csv
dfX.to_csv('X_processed.csv', index=False)
dfY.to_csv('y_processed.csv', index=False)

#save here
#df.to_csv('pokemon_data_encoded.csv', index=False)
