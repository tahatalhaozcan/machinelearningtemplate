# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 16:42:46 2023

@author: tahat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


adds = pd.read_excel('Ads_CTR_Optimisation.xlsx')
print(adds)

N = 10000
d = 10
oduller = [0] *d
tiklamalar = [0] *d
toplam = 0
secilenler = []
for n in range(0,N):
    ad = 0 #secilen ilan 
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ort = oduller[i]/tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb = ort + delta
        else:
            ucb = N * 10
        if max_ucb< ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+1
    odul = adds.values[n,ad]
    oduller[ad] = oduller[ad]+odul
    toplam = toplam + odul

print(toplam)

plt.hist(secilenler)
plt.show()