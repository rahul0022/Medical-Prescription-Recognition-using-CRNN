import numpy as np
import cv2
import os
import string
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, LeakyReLU, Dropout 
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from spellchecker import SpellChecker

class spell(SpellChecker):
    
    def __edit_distance_alt(self, words):
        tmp = [
            w if self._case_sensitive else w.lower()
            for w in words
            if self._check_if_should_check(w)
        ]
        return [e2 for e1 in tmp for e2 in self.known(self.edit_distance_1(e1))]

    def candidates(self, word):
        res = [x for x in self.edit_distance_1(word)]
        tmp = self.known(res)
        if tmp:
            return tmp
        if self._distance == 2:
            tmp = self.known([x for x in self.__edit_distance_alt(res)])
            if tmp:
                return tmp
        return {word}

 

def encode_to_labels(txt):
    
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))
        
    return dig_lst

def process(img):
   
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    img = img.astype('float32')
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)     
    img = img / 255
    
    return img

def make_pred(imagename):

    max_label_len = 0
    char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 


    inputs = Input(shape=(32,128,1))
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)    
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)    
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)    
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)    
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)    
    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)    
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)    
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)    
    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
    act_model = Model(inputs, outputs)


    act_model.load_weights("sgdo-20000r-50e-16005t-1781v.hdf5")

    images = []
    sp = spell(language=None)
    sp.word_frequency.load_text_file("./names.txt")
    try:
        img = cv2.imread(str(imagename), cv2.IMREAD_GRAYSCALE)
        
    except:
        return "image cannot be opened"
    print(imagename, img[100,100])
    img = process(img)
    images.append(img)
    images = np.asarray(images)
    prediction = act_model.predict(images)
    decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]

    out = K.get_value(decoded)
    letters = ''
    for i, x in enumerate(out):
        #print("predicted text = ", end = '')
        letters = ''
        for p in x:
            if int(p) != -1:
                letters+=char_list[int(p)]
        corrected  = sp.correction(letters)
        return corrected


if __name__=='__main__':

    #img = cv2.imread('./5.png', cv2.IMREAD_GRAYSCALE)
    print(make_pred('./upload/5.png'))














