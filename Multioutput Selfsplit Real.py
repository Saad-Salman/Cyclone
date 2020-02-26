# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 20:30:57 2019

@author: Saad
"""

"Import libraries"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"Import Data"
Data = pd.read_excel(r'C:\Users\Saad\Desktop\Data\woc_revised.xlsx')
Xtrain = np.array(Data.iloc[:,[0,1,2,3,4,5]].values)
ytrain = np.array(Data.iloc[:,[6,7]].values)
#print(Xtrain)
#print(ytrain)
"Standardizing"
standard_X = StandardScaler()
standard_y = StandardScaler()
standard_X.fit(Xtrain)
Xtrain = standard_X.transform(Xtrain)
ytrain = standard_y.fit_transform(ytrain)

from sklearn.neural_network import MLPRegressor

"Test Data Set"
"Import Data"
DataT = pd.read_excel(r'C:\Users\Saad\Desktop\Data\Test_set.xlsx')
Xtest1 = np.array(DataT.iloc[0:10,[0,1,2,3,4,5]].values)
ytest1 = np.array(DataT.iloc[0:10,[6,7]].values)



#print(Xtest)
#print(ytest)
Xtest1 = standard_X.transform(Xtest1)
ytest1 = standard_y.transform(ytest1)


'H_L_N = 10'
'R_S = 51'

Multiresult1 = MLPRegressor(hidden_layer_sizes=(10,), random_state=51, max_iter=3000, warm_start=True).fit(Xtrain, ytrain).predict(Xtest1)
Multiresulttrain = MLPRegressor(hidden_layer_sizes=(10,), random_state=51, max_iter=3000, warm_start=True).fit(Xtrain, ytrain).predict(Xtrain)

"R2 and Pearsonr"
#from sklearn.metrics import r2_score
from scipy.stats import pearsonr
print("Test")
#print(r2_score(ytest1[:,0],Multiresult1[:,0]))
#print(r2_score(ytest1[:,1],Multiresult1[:,1]))
print(pearsonr(ytest1[:,0],Multiresult1[:,0])[0])
print(pearsonr(ytest1[:,1],Multiresult1[:,1])[0])
print('\n')
print("Train")
#print(r2_score(ytrain[:,0],Multiresulttrain[:,0]))
#print(r2_score(ytrain[:,1],Multiresulttrain[:,1]))
print(pearsonr(ytrain[:,0],Multiresulttrain[:,0]))
print(pearsonr(ytrain[:,1],Multiresulttrain[:,1]))


