

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:05:03 2023

@author: tahat
"""

import pandas as pd
import numpy as np



yorumlar = pd.read_excel('Kitap1.xlsx')
begenme = yorumlar[['Liked']]


begenme.isnull().values.any() #hiç eksik gözlem var mı?
 #------PREPROCESSING-------#
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
baglaclar =nltk.download('stopwords')
ps = PorterStemmer()

derlem = []
for i in range(1000): #bütün yorumlar için döngü oluşturuyoruz. 
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i]) #noktalama işaretlerini temizleme
    yorum = yorum.lower() #bütün harfleri küçük yapma
    yorum = yorum.split() #kelimeleri boşluklardan parçalama ve listeye atama
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #kelimenin gövdesini alma. bu kelimeler bir kümeye atanıyor. 
    yorum = ' '.join(yorum)
    derlem.append(yorum)

#----------FEATURE EXRTACTION(BOW)-----------#
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(derlem).toarray() #bağımsız değişken
y = yorumlar.iloc[:,1].values #bağımlı değişken


#------MACHINE LEARNING----------#
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20, random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)
pred = gnb.predict(xtest)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,pred)
print(cm)


