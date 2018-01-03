# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
#from nltk.tokenize import RegexpTokenizer


origin_label = []
sents = []

with open('origin_data/tabcore_ENG.txt') as f:
    for line in f:
        temp = line.split('\t')
        origin_label.append(temp[0])
        sents.append(temp[1:])

origin_label = np.array(origin_label)
sents = np.array(sents)

encoder = LabelEncoder()
label = encoder.fit_transform(origin_label)
label = label.reshape((len(label),1))

word_dict = {}
x_numerical = []
count = 1
for sen in sents:
    temp = []
    tokens = sen[0].split(' ')
    for word in tokens:
        
        if word not in word_dict:
            word_dict[word] = count
            count += 1
        temp.append(word_dict[word])
    x_numerical.append(temp)
            
x_padded = pad_sequences(x_numerical,maxlen=10,padding='post',truncating='post')

data = np.hstack((x_padded,label))
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]
np.save('data_enx',X)
np.save('data_eny',Y)

