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
#from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
import time
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D

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
batch_size = 30
num_epochs = 40 
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
########################################################################################################################################
input_shape=(224, 224, 3)
nClasses=10 




image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('activation_49').output

x = AveragePooling2D(pool_size=(4,4),strides=2)(last_layer
)
x= Flatten(name='flatten')(x)
x = Dense(64, activation='relu', name='fc1')(x)
x = Dense(32, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()


# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, Y_train, batch_size, num_epochs, verbose=1, validation_data=(X_test, Y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, Y_test, batch_size, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


model_json = custom_vgg_model2.to_json()
with open("Resnet_Live_Chl.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
custom_vgg_model2.save_weights("Resnet_Live_Chl.h5")
print("Saved model to disk")

