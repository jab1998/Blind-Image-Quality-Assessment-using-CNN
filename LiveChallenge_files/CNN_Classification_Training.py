#########################################################################################################################################
import scipy.io as si
import numpy as np
import cv2
import os
import func as f
import ggd
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split

from keras.models import Model # basic class for specifying and training a neural network
#from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
#X_train= hkl.load( 'X_train.hkl' )
X_test= hkl.load( 'X_test.hkl' )

Y_train= hkl.load( 'Y_train.hkl' )
Y_test= hkl.load( 'Y_test.hkl' )

Y_traintarget= hkl.load( 'Y_traintarget.hkl' )
Y_testtarget= hkl.load( 'Y_testtarget.hkl' )
print ("Files Loaded...!!!")

#num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = 10
batch_size = 30
num_epochs = 20 
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
def createModel():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))


    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model

json_file = open('modelown1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("modelown1.h5")

#model=get_model()
print("Loaded model from disk")
'''
model=createModel()
model.load_weights("modelown1.h5")
#model.load_weights("modelnew1.h5")'''
adm=optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print model.summary()




#model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)

#print (model.evaluate(X_train, Y_train))
print (model.evaluate(X_test, Y_test))

#tr=model.predict(X_train)
te=model.predict(X_test)
k1=np.argmax(te, axis=1)
k2=np.argmax(Y_test, axis=1)
count=0
for i in range(len( k1)):
	print (k1[i],k2[i])
	if k1[i]==k2[i]:
		count=count+1

print count,len(k1) 
########################################################################################################################################
'''
model_json = model.to_json()
with open("modelown1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelown1.h5")
print("Saved model to disk")
'''
########################################################################################################################################


