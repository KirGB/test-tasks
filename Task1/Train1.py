import os
import cv2
import glob
import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
from PIL import Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

num_classes=67
img_width = 200
img_height = 200
# model
base = VGG16(include_top = False, weights = 'imagenet', input_shape = (img_width,img_height,3))
model = Sequential()
model.add(base)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.summary()
train_root='../input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images'
batch_size = 128
nb_epochs = 50

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1) 

train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') 

validation_generator = train_datagen.flow_from_directory(
    train_root, 
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') 

fitting = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs)
model.save('task1.h5')
model.save_weights('task1_weights.h5')