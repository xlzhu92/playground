# -*- coding: utf-8 -*-


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Concatenate,Input,Dense,Embedding,Conv1D,MaxPooling1D,BatchNormalization,Dropout,Flatten,AveragePooling1D
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

def sequentialCNN():
    model = Sequential()
    model.add(Embedding(voc_len+1,150,input_length=30,embeddings_initializer='he_uniform'))
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
    model.save('cnn_parallel_zh.h5')
    
def parralleCNN():
    inputs = Input(shape=(None,))
    embed = Embedding(voc_len+1,150,input_length=30,embeddings_initializer='he_uniform')(inputs)
    
    stream1 = Conv1D(32,5,activation='relu',padding='same')(embed)
    stream1 = MaxPooling1D(pool_size=2)(stream1)
    stream2 = Conv1D(32,4,activation='relu',padding='same')(embed)
    stream2 = MaxPooling1D(pool_size=2)(stream2)
    stream3 = Conv1D(32,3,activation='relu',padding='same')(embed)
    stream3 = MaxPooling1D(pool_size=2)(stream3)
    stream4 = Conv1D(32,2,activation='relu',padding='same')(embed)
    stream4 = MaxPooling1D(pool_size=2)(stream4)
    stream5 = MaxPooling1D(pool_size=2)(embed)
    stream5 = Conv1D(32,1,activation='relu',padding='same')(stream5)
    
    merged = Concatenate(axis=1)([stream1,stream2,stream3,stream4,stream5])
    merged = Flatten()(merged)
    merged = Dropout(0.3)(merged)
    out = BatchNormalization()(merged)
    out = Dense(128,activation='relu')(out)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    #out = Dense(128,activation='relu')(out)
    #out = Dropout(0.3)(out)
    #out = BatchNormalization()(out)
    out = Dense(128,activation='relu')(out)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    out = Dense(class_len,activation='softmax')(out)
    
    model = Model(inputs,out)
    print (model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=200,epochs=10,validation_data=(X_test,Y_test))
    model.save('cnn_parallel_zh.h5')
    
parralleCNN()
'''
pred = model.predict(X_test)


def onehot_decode(seq):
    res = []
    for row in seq:
        temp = np.argmax(row)
        res.append(temp)
    return res 
pred = onehot_decode(pred)
'''