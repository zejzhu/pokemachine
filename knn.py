import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#TODO: make preprocess steps into a function to be used by evaluator
#OR make each model able to return its own evaluations (inefficient but functional)

#load csv files, flatten y to 1d arr
x = pd.read_csv('processed_x.csv')
y = pd.read_csv('processed_y.csv').values.ravel()

#split data 80/20
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

#fit scaler to train, transform both train and test the same way
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#minmax 0-1
minmax = MinMaxScaler
minmax.fit(xtrain)
xtrain = minmax.transform(xtrain)
xtest = minmax.transform(xtest)

#train and predict
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)

train_ypred = knn.predict(xtrain)
test_ypred = knn.predict (xtest)

#evaluate accuracy
trainacc = accuracy_score(train_ypred, ytrain)
testacc = accuracy_score(test_ypred, ytest)

print(f"Training Accuracy: {trainacc:.4f}")
print(f"Test Accuracy: {testacc:.4f}")
