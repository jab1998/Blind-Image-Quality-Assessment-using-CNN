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
#Labels=si.loadmat('./Data/AllMOS_release.mat')['AllMOS_release'][0]

Image_Names=si.loadmat('./Data/AllImages_release.mat')['AllImages_release']
#print np.array([cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])).shape for i in range(1169)])

#Images= np.array([cv2.resize(cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])),(500,500)) for i in range(len(Labels))])
Labels=hkl.load(  'Y_testtarget1.hkl' )
Images =hkl.load(  'X_TestImages.hkl' )
NSStrainfeatures= hkl.load(  'NSStrainfeatures1.hkl' )
NSStestfeatures= hkl.load(  'NSStestfeatures1.hkl' )
print ("Stage:1 Read all the Images")
print 
#########################################################################################################################################

print (len(Labels),Images.shape)
'''
j=0
for Y in tqdm(Labels):
	Y_traini= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		Y_train=Y_traini
		j=1
	else:
		Y_train=np.append(Y_train,Y_traini,axis=0)
'''
json_file = open('modelown1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("modelown1.h5")

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

window_shape = (224, 224, 3)
step=38
from scipy.stats import mode
count=0
j=0
for a in tqdm(range(len(Labels))):
		B = view_as_windows(Images[a], window_shape,step=step)
		patch=B.reshape(B.shape[0]*B.shape[1],224,224,3)
		patchlabel=np.array(patch.shape[0]*[Labels[a]])
		patch = patch.astype('float32')
		print np.max(patch)
		patch /= np.max(patch)
		probi=model.predict(patch)
		he=mode(np.argmax(probi, axis=1))[0][0]
		ti=np.array([[0]*10])
		ti[0][he]=1
		if j==0:
			t=ti
			j=1
		else:
			t=np.append(t,ti,axis=0)
		#he=mode(np.argmax(t, axis=1))[0][0]
		#hi= np.argmax(Y_train[a])
		#if he == hi:
		#	count=count+1
		#print "******************************"

#print count,len(Labels)
print t.shape	
#SVM_Train_Features=np.concatenate((NSStrainfeatures,t),axis=1)
SVM_Test_Features=np.concatenate((NSStestfeatures,t),axis=1)
#SVM_Train_Features = SVM_Train_Features.astype('float32')
SVM_Test_Features = SVM_Test_Features.astype('float32')
#Y_traintarget = Y_traintarget.astype('float32')
#print (type(SVM_Train_Features[0]))
#print (type(SVM_Test_Features[0]))

print ("SVM Training")

print 

clf2 = joblib.load('SVR.joblib') 

k=clf2.predict(SVM_Test_Features)
#print k[0:50]
#print Y_testtarget[0:50]
print spearmanr(Labels,k)
print pearsonr(Labels,k)
'''
print (patch.shape)
print (patchlabel.shape)	
print ("Stage2: Completed (Patch Extraction)")
print



#model=get_model()
k=model.predict(patch)
print("Loaded model from disk")
from scipy.stats import mode
for i in range(0,1216,64):
	print mode(np.argmax(k[i:i+64], axis=1))
	print mode(np.argmax(Y_train[i:i+64],axis=1))
	print "**********************"
'''
