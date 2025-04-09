import pandas as pd
from sklean.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#load csv files, flatten y to 1d arr
x = pd.read_csv('x_processed.csv')
y = pd.read_csv('y_processed.csv').values.ravel()

#split data 80/20
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

#fit scaler to train, transform both train and test the same way
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

dt = DecisionTreeClassifier()
dt.fit(xtrain, ytrain)