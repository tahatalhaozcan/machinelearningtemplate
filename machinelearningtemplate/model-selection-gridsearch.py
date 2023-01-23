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



from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5], 'kernel':['linear']},
     {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma':[0.5,0.1,0.01]}]
gs = GridSearchCV(estimator=classifier, param_grid=p,scoring='accuracy',cv = 4,n_jobs=-1)
grid_search = gs.fit(Xtrain, ytrain)
thebestscore = grid_search.best_score_
thebestparams = grid_search.best_params_

print(thebestscore)
print(thebestparams)
