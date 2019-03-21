#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:16:25 2019

@author: rutvik
"""

#Data Preprocessing


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Bidirectional,LSTM,GlobalMaxPool1D,Dense
import pandas as pd
import numpy as np
import os

#DATA LOADING AND PREPROCESSING

dataset = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv')

sentences  = dataset['comment_text'].values

possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']


target = dataset[possible_labels].values


MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 100
EMB_DIM = 100
VAL_SPLT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

tok = Tokenizer(num_words=MAX_VOCAB_SIZE,lower=True)
tok.fit_on_texts(sentences)
seq = tok.texts_to_sequences(sentences)

word2idx = tok.word_index

data = pad_sequences(seq, maxlen=MAX_SEQ_LEN)



#LOADING EMBEDDINGS


word2vec = {}

with open(os.path.join('./glove.6B/glove.6B.100d.txt')) as f :
    
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vec
        
print(len(word2vec))


#Embedding Matrix and Embedding Layer initialization


num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)

embedding_matrix = np.zeros((num_words,EMB_DIM))

for word,i in word2idx.items():
    if i<MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(input_dim=num_words,output_dim=EMB_DIM,weights = [embedding_matrix],input_length=MAX_SEQ_LEN,trainable = False)


#Building the model

input_ =  Input(shape=(MAX_SEQ_LEN,))
x = embedding_layer(input_)
#x = LSTM(15,return_sequences =True)(x)
x = Bidirectional(LSTM(15,return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels),activation="sigmoid")(x)

model = Model(input_,output)
model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy']
        )

#Training Model

l = model.fit(x = data,
              y = target,
              batch_size=128,
              epochs=EPOCHS,
              validation_split=VAL_SPLT)

import matplotlib.pyplot as plt

plt.plot(l.history['loss'],label='loss')
plt.plot(l.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(l.history['acc'],label='acc')
plt.plot(l.history['val_acc'],label='val_acc')
plt.legend()
plt.show()


#Testing with custom input

sample = "you are a looser"

samp = []
samp.append(sample)

sample_seq = tok.texts_to_sequences(samp)

sample_seq = pad_sequences(sample_seq,maxlen=MAX_SEQ_LEN)

res = model.predict(sample_seq)

for i in range(len(res[0])):
    if res[0][i] > 0.5:
        print(possible_labels[i])
