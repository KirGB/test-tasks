import os
import pandas as pd
import numpy as np 
import tensorflow as tf 
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image

model1=load_model('../input/task1/task1.h5')
model1.summary()
model2=load_model('../input/task1-2/task1_2.h5')
model2.summary()
train_root='../input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images'
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    subset='training')
classes=list(train_generator.class_indices.keys())

# test solution
test_images = os.listdir(f'../input/test-images/Test_images')
for i in range(len(test_images)):
    print(test_images[i])
    test_image1 = image.load_img(f'../input/test-images/Test_images/{test_images[i]}', target_size = (200, 200))
    test_image1 = image.img_to_array(test_image1)/255
    test_image1 = np.expand_dims(test_image1, axis = 0)
    result1 = model1.predict(test_image1)
    
    test_image2 = image.load_img(f'../input/test-images/Test_images/{test_images[i]}', target_size = (224, 224))
    test_image2 = image.img_to_array(test_image2)/255
    test_image2 = np.expand_dims(test_image2, axis = 0)
    result2 = model2.predict(test_image2)
    
    for i in range (0,len(classes)):
        
        if result1[0][i]>0.5:
            class1=classes[i]
        if result2[0][i]>0.5:
            class2=classes[i]
            
    print('Model 1 Prediction is',class1)
    print('Model 2 Prediction is',class2)
