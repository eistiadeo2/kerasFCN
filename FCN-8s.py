from __future__ import print_function


from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Permute, Reshape
from keras.layers.merge import add
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


from keras.optimizers import SGD


from keras.utils import plot_model

import cv2, numpy as np
import sys


nb_classes = 21
nb_epoch = 5
batch_size = 32
img_rows = 224
img_cols = 224
samples_per_epoch = 1190
nb_val_samples = 170

def create_model():

    #(samples, channels, rows, cols)
    input_img = Input(shape=(3, img_rows, img_cols))
    #(3*224*224)
    x = Conv2D(64, 3, strides=(1, 1), activation='relu',padding='same')(input_img)
    x = Conv2D(64, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(64*112*112)
    x = Conv2D(128, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(128, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(128*56*56)
    x = Conv2D(256, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(256, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(256*28*28)

    #split layer
    p3 = x
    p3 = Conv2D(nb_classes, 1, strides=(1, 1),activation='relu')(p3)
    #(21*28*28)

    x = Conv2D(512, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(512, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*14*14)

    #split layer
    p4 = x
    p4 = Conv2D(nb_classes, 1, strides=(1, 1), activation='relu')(p4)
    p4 = UpSampling2D(size=(2, 2))(p4)
    #(21*28*28)


    x = Conv2D(512, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(512, 3, strides=(1, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*7*7)

    p5 = x
    p5 = Conv2D(nb_classes, 1, strides=(1, 1), activation='relu')(p5)
    p5 = UpSampling2D(size=(4, 4))(p5)
    #(21*28*28)

    # merge scores
    merged = add([p3, p4, p5])
    x = Conv2DTranspose(nb_classes, 16, strides=(8, 8), padding='same')(merged)
    x = Flatten()(x)
    out = Activation("softmax")(x)
    #(21,224,224)
    model = Model(input_img, out)
    return model

model = create_model()

model.summary()


plot_model(model, to_file='FCN-8s_model.png', show_shapes=True)
