# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:27:38 2023

@author: Eason
"""
import scipy.io
import tensorflow as tf
import os
from PIL import ImageOps
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Dense, Flatten

import numpy as np
import matplotlib.pyplot as plt


def build_generator(input_shape):
    inputs = Input(input_shape)
    leaky_relu = tf.keras.layers.LeakyReLU()

    # Encoder (contraction) path
    c1 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(inputs)
    c1 = BatchNormalization()(c1)
    c1 = leaky_relu(c1)
    c1 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = leaky_relu(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(p1)
    c2 = BatchNormalization()(c2)
    c2 = leaky_relu(c2)
    c2 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = leaky_relu(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(p2)
    c3 = BatchNormalization()(c3)
    c3 = leaky_relu(c3)
    c3 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(c3)
    c3 = BatchNormalization()(c3)
    c3 = leaky_relu(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottom layer
    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = leaky_relu(c4)

    # Decoder (expansion) path
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Conv2D(256, (2, 2), padding='same')(u5)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(u5)
    c5 = BatchNormalization()(c5)
    c5 = leaky_relu(c5)
    c5 = Conv2D(256, (3, 3), padding='same', dilation_rate=1)(c5)
    c5 = BatchNormalization()(c5)
    c5 = leaky_relu(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(128, (2, 2), padding='same')(u6)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(u6)
    c6 = BatchNormalization()(c6)
    c6 = leaky_relu(c6)
    c6 = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(c6)
    c6 = BatchNormalization()(c6)
    c6 = leaky_relu(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(64, (2, 2), padding='same')(u7)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(u7)
    c7 = BatchNormalization()(c7)
    c7 = leaky_relu(c7)
    c7 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = leaky_relu(c7)
    c7 = Conv2D(1, (1, 1), padding='same', dilation_rate=2)(c7)
    c7 = Flatten()(c7)

    # Output layer
    outputs = Dense(120, activation='softmax')(c7)
    

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, beta_1=0.6, clipnorm=0.01, epsilon=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model

def load(path,target_size):
  ret_data = []
  ret_label = []
  ret_name = []
  count = 0

  for i in path: #path is a list of folder path
    file_list = os.listdir(i)
    # Filter out non-image files (if any)
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in file_list if os.path.splitext(f)[1].lower() in img_extensions]
    for f in img_files: #f is image name is the folder
        img_path = os.path.join(i, f)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        
        # Original image
        img_array = np.array(img)
        label = np.zeros(120)
        label[count] = 1 #Create one-hot encoding label
        ret_data.append(img_array)
        ret_label.append(label)
        ret_name.append(f)      
    count += 1
  

        
  ret_data , ret_label = np.array(ret_data),np.array(ret_label)
  return ret_data,ret_label,ret_name




if __name__ == "__main__":
    #load .mat file
    file_name  = "C:/Users/88696/Desktop/三下課程/影像處理/project3/Nevada.mat"
    #mat_data = scipy.io.loadmat(file_name)
    #hyperspectral_data = mat_data['X']
    
    #length , width ,height = hyperspectral_data.shape
    #target_size = (length,width)
    
    #initailize the model
    model = build_generator((256,256,3))
    
    
            
    
