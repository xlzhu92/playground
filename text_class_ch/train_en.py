# -*- coding: utf-8 -*-
from keras import regularizers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Input,Concatenate,Dense,Embedding,Conv1D,MaxPooling1D,BatchNormalization,Dropout,Flatten,AveragePooling1D
import numpy as np

X = np.load('data_enx.npy')
Y = np.load('data_eny.npy')

Y_cate = to_categorical(Y)
class_len = Y_cate.shape[1]
voc_len = np.amax(X)

def SequentialCNN():
    model = Sequential()
    model.add(Embedding(voc_len+1,150,input_length=10,embeddings_initializer='he_uniform'))
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
    #model.add(BatchNormalization())
    #model.add(Dense(128,activation='relu'))
    #model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(class_len,activation='softmax'))
    print (model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X,Y_cate,batch_size=200,epochs=30,validation_split=0.2)
    model.save('cnn_sequential_en.h5')
    
    
def ParallelCNN():
    set_l2 = 0.01
    inputs = Input(shape=(None,))
    embed = Embedding(voc_len+1,150,input_length=10,embeddings_initializer='he_uniform')(inputs)
    
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
    
    merged = Conv1D(64,5,activation='relu',padding='valid',
                    kernel_regularizer=regularizers.l2(set_l2))(merged)
    merged = MaxPooling1D(pool_size=5)(merged)
    merged = Conv1D(64,3,activation='relu',padding='valid',
                    kernel_regularizer=regularizers.l2(set_l2))(merged)
    merged = MaxPooling1D(pool_size=2)(merged)
    merged = Flatten()(merged)
    #merged = Dropout(0.3)(merged)
    out = BatchNormalization()(merged)
    #out = Dense(128,activation='relu')(out)
    #out = Dropout(0.3)(out)
    #out = BatchNormalization()(out)
    #out = Dense(128,activation='relu')(out)
    #out = Dropout(0.3)(out)
    #out = BatchNormalization()(out)
    #out = Dense(128,activation='relu')(out)
    out = Dropout(0.3)(out)
    #out = BatchNormalization()(out)
    out = Dense(class_len,activation='softmax',
                kernel_regularizer=regularizers.l2(set_l2))(out)
    
    model = Model(inputs,out)
    print (model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X,Y_cate,batch_size=200,epochs=20,validation_split=0.2)
    model.save('cnn_parallel_en.h5')

ParallelCNN()