# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:23:54 2023

@author: tahat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


veriler = pd.read_excel('eksikveri.xlsx')


x = veriler.iloc[4:, 1:4].values
y = veriler.iloc[4:,4:].values
print(x)
print(y) 

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(xtrain) #eğitme
Xtest = sc.transform(xtest)#eğitimi kullanma


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticr  = LogisticRegression()
logisticr.fit(Xtrain,ytrain)
ypred = logisticr.predict(Xtest)
print(ypred)
from sklearn.metrics import confusion_matrix #karmaşıklık matrixi
cm = confusion_matrix(ytest, ypred) #gerçek değerler ve tahmin değerleri
print('LOGISTIC REGRESSION')
print(cm)

#KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()#kernel=rbf parametresi verirse kernel trick uygulayabiliriz.
knn.fit(Xtrain,ytrain)
y2knn_pred = knn.predict(Xtest)
cm2 = confusion_matrix(ytest, y2knn_pred)
print(('KNN'))
print(cm2)


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Xtrain, ytrain)
y3gnb_pred = gnb.predict(Xtest)
cm3 = confusion_matrix(ytest, y3gnb_pred)
print('GNB')
print(cm3)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(Xtrain,ytrain)
y4dtc_pred = dtc.predict(Xtest)
cm4 = confusion_matrix(ytest, y4dtc_pred)
print('DTC')
print(cm4)


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 5 ,criterion='entropy')
rfc.fit(Xtrain,ytrain)
y5rfc_pred = rfc.predict(Xtest)

cm5 = confusion_matrix(ytest, y5rfc_pred)
print('RFC')
print(cm5)



#ROC EĞRİSİ
y_proba = rfc.predict_proba(Xtest)
print('PROBABILITY')
print(y_proba)
from sklearn import metrics
fpr, tpr, thold =metrics.roc_curve(ytest, y_proba[:,0],pos_label='e')
print(ytest)
print('ROC CURVE')
print(fpr)
print(tpr)
print(thold)