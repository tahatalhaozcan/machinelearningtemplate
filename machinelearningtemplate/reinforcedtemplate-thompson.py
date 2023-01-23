# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:51:47 2023

@author: tahat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random



adds = pd.read_excel('Ads_CTR_Optimisation.xlsx')

N = 10000
d = 10
toplam = 0
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #secilen ilan 
    max_ts = 0
    for i in range(0,d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if rasbeta>max_ts:
            max_ts = rasbeta
            ad = i
        
    secilenler.append(ad)
    odul = adds.values[n,ad]
    if odul == 1:
        birler[ad] = birler[ad] +1
    else:
        sifirlar[ad] = sifirlar[ad] +1
        

    toplam = toplam + odul

print(toplam)

plt.hist(secilenler)
plt.show()