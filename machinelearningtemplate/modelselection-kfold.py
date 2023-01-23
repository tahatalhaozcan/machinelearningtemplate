# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:47:44 2023

@author: tahat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("Social_Network_Ads.csv")
X= veriler.iloc[:,[2,3]].values
Y = veriler.iloc[:,4].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.25,random_state=0)
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

Xtrain = sc.fit_transform(xtrain)
Xtest = sc.fit_transform(xtest)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state= 0)
classifier.fit(Xtrain, ytrain)

ypred =classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)
print(cm)

from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator = classifier, X = Xtrain, y=ytrain,cv = 4)
print(basari.mean())