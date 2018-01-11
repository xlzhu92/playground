# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import jieba

np.random.seed(7)

train = pd.read_csv('origin_data/TrainSet-eCarX-171019.txt',header=None, delim_whitespace=True,encoding='gbk')
test = pd.read_csv('origin_data/TestSet-eCarX-171019.txt',header=None, delimiter='#',encoding='gbk')
del test[1]
del test[3]

sens = pd.concat([train[1],test[0]])
labs = pd.concat([train[0],test[2]])

flag = len(train) #flag of train/test margin

seg_sens = []
for sen in sens:
    res = jieba.lcut(sen,cut_all=False)
    seg_sens.append(res)
    
encoder = LabelEncoder()
label = encoder.fit_transform(labs)
label = label.reshape((len(label),1))

char_dict = {}
count = 1
for row in seg_sens:
    for word in row:
        if word not in char_dict:
            char_dict[word] = count
            count += 1
            
x_numerical = []
for row in seg_sens:
    temp = []
    for word in row:
        temp.append(char_dict[word])
    x_numerical.append(temp)

x_padded = pad_sequences(x_numerical,maxlen=20,padding='post',truncating='post')

data = np.hstack((x_padded,label))

Train = data[:flag]
Test = data[flag:]


np.random.shuffle(Train)
np.random.shuffle(Test)

X_train = Train[:,:-1]
Y_train = Train[:,-1]
X_test = Test[:,:-1]
Y_test = Test[:,-1]

np.save('jieba_data/trainx',X_train)
np.save('jieba_data/trainy',Y_train)
np.save('jieba_data/testx',X_test)
np.save('jieba_data/testy',Y_test)
