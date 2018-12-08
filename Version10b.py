#########################################################################################################################################
import scipy.io as si
import numpy as np
import cv2
import os
import func as f
import ggd
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from keras.layers import Input, Convolution2D, MaxPooling2D,AveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model # basic class for specifying and training a neural network
from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.models import model_from_json

X_train= hkl.load( 'X_train.hkl' )
X_test= hkl.load( 'X_test.hkl' )

Y_train= hkl.load( 'Y_train.hkl' )
Y_test= hkl.load( 'Y_test.hkl' )

Y_traintarget= hkl.load( 'Y_traintarget.hkl' )
Y_testtarget= hkl.load( 'Y_testtarget.hkl' )
print ("Files Loaded...!!!")

num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = 10
batch_size = 64
num_epochs = 25 
kernel_size = 3 
pool_size = 2 
conv_depth_1 = 32 
conv_depth_2 = 64
conv_depth_3 = 128
conv_depth_4 = 256
conv_depth_5 = 512 
drop_prob_1 = 0.5
drop_prob_2 = 0.25
hidden_size = 512
print ("Training About to start")
print
#########################################################################################################################################

def get_model():
    input_tensor = Input(shape=(224, 224, 3)) 
    base_model = applications.VGG16(weights='imagenet',include_top=False, input_tensor=input_tensor)
    for layer in range(len(base_model.layers)):
        base_model.layers[layer].trainable=False
    x = base_model.layers[-1].output
    x = AveragePooling2D(pool_size=(pool_size, pool_size),strides=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    drop = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(drop)
    #drop = Dropout(drop_prob_1)(x)
    x = Dense(num_classes, activation='softmax')(drop)
    updatedModel = Model(base_model.input, x)
    return  updatedModel

json_file = open('model5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model5.h5")
print("Loaded model from disk")

#model=get_model()

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

print model.summary()




model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.15)

print (model.evaluate(X_train, Y_train, verbose=1))
print (model.evaluate(X_test, Y_test, verbose=1))

########################################################################################################################################

model_json = model.to_json()
with open("model6.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model6.h5")
print("Saved model to disk")

########################################################################################################################################


