import pandas as pd
from sklearn.svm import SVC
import time
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from preprocessing import *
import matplotlib.pyplot as plt
import pickle

# data = pd.read_csv('House_Data_Classification.csv')
# data = splitData(data)
# data.dropna(axis = 1, how = 'any', inplace = True)
# # data1 = data[['LotArea', 'Street', 'MSSubClass', 'MasVnrType', 'SalePrice']]
# # data1.dropna(how = 'any', inplace = True)
# columns = ('Street', 'LotShape', 'Utilities', 'MSZoning', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'f1', 'f2', 'f3', 'PriceRate')
# data = Feature_Encoder(data, columns)
# data.drop(data.iloc[:, 30:-4], inplace = True, axis = 1)
# X = data.drop('PriceRate', axis=1)
# print(X)
# Y = data.iloc[:, -4:-3]
# print(Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
# # SVM Model using rbf kernel
# startTrain = time.time()
# model1 = SVC(kernel='rbf', C=1.0)
# model1.fit(X_train, Y_train)
# endTrain = time.time()
# startTest = time.time()
# accuracy = model1.score(X_test, Y_test)
# endTest = time.time()
# print('Model #1 Accuracy: ', accuracy)
# print('Model #1 Training Time: ', endTrain - startTrain, ' seconds')
# print('Model #1 Testing Time: ', endTest - startTest, ' seconds')
# print(' ')
#
# #Decision Tree
# startTrain = time.time()
# model2 = tree.DecisionTreeClassifier(max_depth=3)
# model2.fit(X_train, Y_train)
# endTrain = time.time()
# startTest = time.time()
# accuracy = model2.score(X_test, Y_test)
# endTest = time.time()
# print('Model #2 Accuracy: ', accuracy)
# print('Model #2 Training Time: ', endTrain - startTrain, ' seconds')
# print('Model #2 Testing Time: ', endTest - startTest, ' seconds')
# print(' ')
#
# startTrain = time.time()
# model3 = LogisticRegression(max_iter=1000).fit(X_train, Y_train)
# endTrain = time.time()
# startTest = time.time()
# accuracy = model3.score(X_test, Y_test)
# endTest = time.time()
# print('Accuracy: ', accuracy)
# print('Training Time: ', endTrain - startTrain, ' seconds')
# print('Testing Time: ', endTest - startTest, ' seconds')

# pickle.dump(model1, open('model1.sav', 'wb'))
# pickle.dump(model2, open('model2.sav', 'wb'))
# pickle.dump(model3, open('model3.sav', 'wb'))

data = pd.read_csv('House_Data_Classification.csv')
data = splitData(data)
data.dropna(axis = 1, how = 'any', inplace = True)
columns = ('Street', 'LotShape', 'Utilities', 'MSZoning', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'f1', 'f2', 'f3', 'PriceRate')
data = Feature_Encoder(data, columns)
data.drop(data.iloc[:, 30:-4], inplace = True, axis = 1)
XTest = data.drop('PriceRate', axis=1)
YTest = data.iloc[:, -4:-3]
model1 = pickle.load(open('model1.sav', 'rb'))
model2 = pickle.load(open('model2.sav', 'rb'))
model3 = pickle.load(open('model3.sav', 'rb'))
print('acc1: ', model1.score(XTest, YTest))
print('acc2: ', model2.score(XTest, YTest))
print('acc3: ', model3.score(XTest, YTest))