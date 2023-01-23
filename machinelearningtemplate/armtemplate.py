# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:46:02 2023

@author: tahat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ecufast = pd.read_excel('islemlistesi.xlsx', header = None)
islem = []

for i in range(0,17):
    islem.append([str(ecufast.values[i,j])for j in range(0,6)]) 
from apyori import apriori
rules = apriori(islem, min_support = 0.1,min_confidence = 0.2, min_lift=3,min_length = 2, max_lenght = 5)
print(list(rules))