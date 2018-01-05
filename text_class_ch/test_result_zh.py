# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


pred = np.loadtxt('predict_zh.txt',dtype=int)
Y_test = np.load('testy.npy')

test = pd.read_csv('origin_data/TestSet-eCarX-171019.txt',header=None, delimiter='#',encoding='gbk')
del test[1]
del test[3]

encoder = LabelEncoder()
encoder.fit(test[2])
trans_pred = encoder.inverse_transform(pred)
trans_y = encoder.inverse_transform(Y_test)

wrong_cases = {}
wrong_count = {}

for index in range(len(trans_pred)):
    if trans_pred[index] != trans_y[index]: 
        wrong_count[trans_y[index]] = wrong_count.get(trans_y[index],0) + 1
        cases_count = wrong_cases.get(trans_y[index],{})
        cases_count[trans_pred[index]] = cases_count.get(trans_pred[index],0) + 1
        wrong_cases[trans_y[index]] = cases_count

        print ('True label:{}\nPred label:{}\n'.format(trans_y[index],trans_pred[index]))