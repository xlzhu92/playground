# -*- coding: utf-8 -*-


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,BatchNormalization,Dropout,Flatten,AveragePooling1D
import numpy as np


X_train = np.load('trainx.npy')
Y_train = np.load('trainy.npy')
X_test = np.load('testx.npy')
Y_test = np.load('testy.npy')

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
voc_len = max(np.amax(X_train),np.amax(X_test))
#voc_len = np.amax(X_train)
class_len = Y_train.shape[1]


model = Sequential()
model.add(Embedding(voc_len+1,100,input_length=30,embeddings_initializer='he_uniform'))
model.add(Conv1D(128,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv1D(128,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
#model.add(Conv1D(64,3,activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2))
#model.add(BatchNormalization())
#model.add(Conv1D(128,2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv1D(64,2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv1D(64,2,activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(class_len,activation='softmax'))
print (model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=200,epochs=15,validation_data=(X_test,Y_test))
model.save('cnn.h5')

pred = model.predict(X_test)


def onehot_decode(seq):
    res = []
    for row in seq:
        temp = np.argmax(row)
        res.append(temp)
    return res 
pred = onehot_decode(pred)
