# -*- coding: utf-8 -*-

from keras.models import Sequential,Model,load_model
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


np.random.seed(7)

'''
train = pd.read_csv('origin_data/TrainSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')
test = pd.read_csv('origin_data/TestSet-eCarX-171019.txt',header=None, delimiter='#',encoding='gbk')
del test[1]
del test[3]

sens = pd.concat([train[1],test[0]])
labs = test[2]

flag = len(train) #flag of train/test margin
       

char_dict = {}
count = 1
for row in sens:
    for char in row:
        if char not in char_dict:
            char_dict[char] = count
            count += 1

x_numerical = []
for row in sens:
    temp = []
    for char in row:
        temp.append(char_dict[char])
    x_numerical.append(temp)

data = pad_sequences(x_numerical,maxlen=30,padding='post',truncating='post')


Train = data[:flag]
Test = data[flag:]



model = load_model('saved_models/cnn_parallel_zh.h5')
layer_name = 'bn_layer'
feature_layer = Model(inputs=model.inputs,
                      outputs=model.get_layer(layer_name).output)
feature_output = feature_layer.predict(Test)
'''

flag1 = 100
flag2 = 200
flag3 = 300

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(feature_output)

plt.plot(reduced_data[:flag1][:,0],reduced_data[:flag1][:,1],'ro')
plt.plot(reduced_data[flag1:flag2][:,0],reduced_data[flag1:flag2][:,1],'bo')
plt.plot(reduced_data[flag2:flag3][:,0],reduced_data[flag2:flag3][:,1],'go')
plt.show()