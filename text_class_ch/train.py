# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,BatchNormalization,Dropout,Flatten

X = np.load('datax.npy')
Y = np.load('datay.npy')
Y = to_categorical(Y)

voc_len = np.amax(X)
class_len = Y.shape[1]



def paral_conv():
    conv1 = Conv1D(64,3,activation='relu')
    


model = Sequential()
model.add(Embedding(voc_len+1,100,input_length=30))
model.add(Conv1D(128,3,activation='relu',padding='same'))
model.add(Conv1D(128,3,activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64,2,activation='relu',padding='same'))
model.add(Conv1D(64,2,activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))

model.add(Dense(class_len,activation='softmax'))
print (model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,batch_size=200,epochs=30,validation_split=0.2)
model.save('cnn.h5')
