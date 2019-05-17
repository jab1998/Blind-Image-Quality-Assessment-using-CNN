#########################################################################################################################################
import scipy.io as si
import numpy as np
import cv2
import os
import func as f
import ggd
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model # basic class for specifying and training a neural network
from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from skimage.util import view_as_windows
from scipy import signal
import math
from keras.models import model_from_json
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr,pearsonr
from scipy.stats import mode

Labels=hkl.load(  'CSIQ_Labels.hkl' )
Images =hkl.load(  'CSIQ_Images.hkl' )
NSStrainfeatures= hkl.load(  'NSStrainfeatures_CSIQ.hkl' )

print ("Stage:1 Read all the Images")
print 
#########################################################################################################################################

print (len(Labels),Images.shape)


json_file = open('Resnet_CSIQ.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("Resnet_CSIQ.h5")

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

window_shape = (224, 224, 3)
step=38

count=0
j=0
for a in tqdm(range(len(Labels))):
		B = view_as_windows(Images[a], window_shape,step=step)
		patch=B.reshape(B.shape[0]*B.shape[1],224,224,3)
		patchlabel=np.array(patch.shape[0]*[Labels[a]])
		patch = patch.astype('float32')
		patch /= np.max(patch)
		probi=model.predict(patch)
		he=mode(np.argmax(probi, axis=1))[0][0]
		ti=np.array([[0]*10])
		ti[0][he]=1
		print ti[0],Labels[a]
		print "*******"
		if j==0:
			t=ti
			j=1
		else:
			t=np.append(t,ti,axis=0)

print t.shape	
SVM_Train_Features=np.concatenate((NSStrainfeatures,t),axis=1)
SVM_Train_Features = SVM_Train_Features.astype('float32')

print ("SVM Training")


clf2 = joblib.load('SVR.joblib') 

k=clf2.predict(SVM_Train_Features)
print spearmanr(Labels,k)
print pearsonr(Labels,k)

