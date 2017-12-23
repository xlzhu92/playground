# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences

raw_data = pd.read_csv('TrainSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')
raw_test = pd.read_csv('TestSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')


encoder = LabelEncoder()
label = encoder.fit_transform(raw_data[0])
label = label.reshape((len(label),1))

char_dict = {}
count = 1
for row in raw_data[1]:
    for char in row:
        if char not in char_dict:
            char_dict[char] = count
            count += 1

x_numerical = []
for row in raw_data[1]:
    temp = []
    for char in row:
        temp.append(char_dict[char])
    x_numerical.append(temp)

x_padded = pad_sequences(x_numerical,maxlen=30,padding='post',truncating='post')

data = np.hstack((x_padded,label))
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

np.save('datax',X)
np.save('datay',Y)
