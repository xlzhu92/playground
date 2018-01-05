# -*- coding: utf-8 -*-

import gensim
import jieba
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences

np.random.seed(7)

train = pd.read_csv('origin_data/TrainSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')
test = pd.read_csv('origin_data/TestSet-eCarX-171019.txt',header=None, delimiter='#',encoding='gbk')
del test[1]
del test[3]

sens = pd.concat([train[1],test[0]])

seg_sens = []

for sen in sens:
    seg_sens.append(jieba.lcut(sen,cut_all=False))
    
model = gensim.models.Word2Vec(seg_sens, size=150)
model.save('trained_wv')