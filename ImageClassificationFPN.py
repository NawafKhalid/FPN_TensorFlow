import numpy as np
import cv2
import os
import random
import shutil
import pandas as pd
import csv
import zipfile
import io
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard # Visualize the model 

import keras
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Input

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
import keras.backend as k

# !pip install keras_retinanet
import keras_retinanet 
from keras_retinanet import layers

from sklearn.utils import shuffle

from PIL import Image as pil_image


# This part can be skipped.
############################################################################################
pickle_in = open("/Users/nawafalageel/NGHA/XFD_test.pickle","rb") # Load the training dataset 
X = pickle.load(pickle_in)
print(X.shape)

pickle_in = open("/Users/nawafalageel/NGHA/yFD_test.pickle","rb") # Load the training labels 
y = pickle.load(pickle_in)
print(len(y))

#Since we know the highest and lowest value we just divided by 255. 
X=np.array(X/255.0) #Normalize the dataset before we feed it to the network. 
y=np.array(y)

############################################################################################


k.clear_session() #Clear keras backend 
try:
  os.mkdir('models') #create folder for saving the trained networks
except:
  pass



# Number of classes
# classes_number = 2 

# Specify input shape for the network
# X.shape[1:] # If you not sure what is the size of your images. 
input_tensor = Input(X.shape[1:]) 
print(X.shape[1:])
 
# Load ResNet101, MobileNetV2, ... ImageNet pre-trained weights
# Visit https://keras.io/api/applications/ for more models.
weight_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False)

# Save the weights
weight_model.save_weights('weights.h5')

# Load the ResNet101 model without weights
base_model = tf.keras.applications.ResNet101(weights=None, include_top=False, input_shape=tuple(X.shape[1:]))


# Load the ImageNet weights on the ResNet101 model
# except the first layer(because the first layer has one channel in our case)
# .load_weights is only 
base_model.load_weights('weights.h5', skip_mismatch=True, by_name=True)



NAME = "ImageClassificationFPN" #Name to our model 

full_name='ResNet101-FPN-fold{}'.format(NAME)

# Path to save the trained models
filepath="models/%s-{epoch:02d}-{val_accuracy:.4f}.hdf5"%full_name

# To be able to open the graph with tensorflow board  
# Trough the terminal make sure that your in the the file that has the logs file 
# and then write "tensorboard --logdir=logs/'modelName'".
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) 

# To stop the trainig process when there is no better results after patience epochs
EarlyStop = tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss', verbose=1, )

# Creating checkpoint to save the best validation accuracy
# And saves the best model weights at some frequency.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list = [tensorboard, checkpoint, EarlyStop]


base_model.summary()


# Create Feature Pyramid Network (FPN)
############################################################################################

# We used some help for writing the Pyramid from the written code on https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/models/retinanet.py
# Set the feature channels of the FPN
# Go check FPN original paper to see why 256, even though it can be any value.
feature_size=256 

# Layers of ResNet101 with different scale features
# From ResNet you can find the names of layers
# You can use base_model.summary() to see the architecture and then extract what layers you want as an output
layer_names = ["conv3_block4_out", "conv4_block11_out", "conv4_block23_out", "conv5_block3_out"] 
layer_outputs = [base_model.get_layer(name).output for name in layer_names]
          
# Features of different scales, extracted from ResNet101
C2, C3, C4, C5 = layer_outputs 

print("Layer outputs = ", layer_outputs)
print("C2 = ", C2)
print("C3 = ", C3)
print("C4 = ", C4)
print("C5 = ", C5)



# To freez all the backbone "base model" model "ResNet"
base_model.trainable = False 

# Except C2, C3, C4, C5
C2.trainable = True
C3.trainable = True
C4.trainable = True
C5.trainable = True


P5         = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
P5         = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

# Concatenate P5 elementwise to C4
P4         = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
P4         = keras.layers.Concatenate(axis=3)([P5_upsampled, P4])
P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
P4         = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P4')(P4)

# Concatenate P4 elementwise to C3
P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
P3 = keras.layers.Concatenate(axis=3)([P4_upsampled, P3])
P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C2])
P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P3')(P3)

# "P6 is obtained via a 3x3 stride-2 conv on C5"
P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
P2 = keras.layers.Concatenate(axis=3)([P3_upsampled, P2])
P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P2')(P2)

#################################################################


# Run classification for each of the generated features from the pyramid
#################################################################
feature1 = Flatten()(P2)
dp1 = Dropout(0.5)(feature1)
preds1 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp1)


feature2 = Flatten()(P3)
dp2 = Dropout(0.5)(feature2)
preds2 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp2)


feature3 = Flatten()(P4)
dp3= Dropout(0.5)(feature3)
preds3 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp3)


feature4 = Flatten()(P5)
dp4 = Dropout(0.5)(feature4)
preds4 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp4)

#################################################################

# Concatenate the predictions(Classification results) of each of the pyramid features
concat=keras.layers.Concatenate(axis=1)([preds1,preds2,preds3,preds4]) 

#Final Classification
out=keras.layers.Dense(1,activation='sigmoid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(concat) 

#Create the Training Model    
model = Model(inputs=base_model.input, outputs=out)

#######################################################


# The model include backbone and FPN
model.summary()



model.compile(
    optimizer=optimizers.Adam(lr=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy'])



model.fit(X, y, batch_size=16, epochs=50, validation_split=0.1, callbacks=callbacks_list)




