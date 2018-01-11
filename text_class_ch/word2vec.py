# -*- coding: utf-8 -*-

import gensim
import jieba
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import multiprocessing

np.random.seed(7)

train = pd.read_csv('origin_data/TrainSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')
test = pd.read_csv('origin_data/TestSet-eCarX-171019.txt',header=None, delimiter='#',encoding='gbk')
del test[1]
del test[3]

sens = pd.concat([train[1],test[0]])

vec_len = 150
'''
seg_sens = []

for sen in sens:
    seg_sens.append(jieba.lcut(sen,cut_all=False))
    
model = gensim.models.Word2Vec(seg_sens, size=vec_len,min_count=1,workers=multiprocessing.cpu_count())
model.save('trained_wv')
'''
model = gensim.models.Word2Vec.load('trained_wv')

max_len = 0
sen_vec = []
for sen in sens:
    sen_cut = jieba.lcut(sen,cut_all=False)
    if len(sen_cut) > max_len:
        max_len = len(sen_cut)
    vec_cut = []
    for word in sen_cut:
        vec_cut.append(model.wv[word])
    sen_vec.append(np.array(vec_cut))
sen_vec = np.array(sen_vec)
        
np.save('wv_data',sen_vec)