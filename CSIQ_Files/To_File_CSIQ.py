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
from keras import optimizers
import hickle as hkl 
from skimage.util import view_as_windows
import pandas as pd
import glob
from tqdm import tqdm

filenames = glob.glob("./CSIQ/dst_imgs/awgn/*.png")
filenames.sort()
Images = [cv2.imread(img) for img in filenames]

folders=["jpeg","jpeg2000","fnoise","blur","contrast"]
i=0

for folder in folders:
	filenames = glob.glob("./CSIQ/dst_imgs/"+folder+"/*.png")
	filenames.sort()
	imagesi = [cv2.imread(img) for img in filenames]
	Images=np.append(Images,imagesi,axis=0)
	print(len(Images))

Labels=pd.read_excel("./CSIQ/labels.xlsx")['Labels'].values
Labels=Labels*100
print (Images.shape)
print (Labels.shape)

print ("Stage:1 Read all the Images")
print 
#########################################################################################################################################

#print (len(Labels),Images.shape)


window_shape = (224, 224, 3)
step=38
i=0
for a in tqdm(range(len(Labels))):
	if i==0:
		B = view_as_windows(Images[a], window_shape,step=step)
		patch=B.reshape(B.shape[0]*B.shape[1],224,224,3)
		patchlabel=np.array(patch.shape[0]*[Labels[a]])
		i=1
	else:
		B = view_as_windows(Images[a], window_shape,step=step)
		patchi=B.reshape(B.shape[0]*B.shape[1],224,224,3)
		patchlabeli=np.array(patchi.shape[0]*[Labels[a]])
		patch=np.append(patch,patchi,axis=0)
		patchlabel=np.append(patchlabel,patchlabeli,axis=0)	

print (patch.shape)
print (patchlabel.shape)	
print ("Stage2: Completed (Patch Extraction)")
print 
#########################################################################################################################################

X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(patch, patchlabel, test_size=0.2, random_state=42)
print (X_train.shape,Y_traintarget.shape)
print (X_test.shape, Y_testtarget.shape)
print ("stage3: Test Train Split Completed")
j=0
for Y in tqdm(Y_traintarget):
	Y_traini= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		Y_train=Y_traini
		j=1
	else:
		Y_train=np.append(Y_train,Y_traini,axis=0)
j=0
for Y in tqdm(Y_testtarget):
	Y_testi= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		Y_test=Y_testi
		j=1
	else:
		Y_test=np.append(Y_test,Y_testi,axis=0)

print (X_train.shape,Y_train.shape)
print (X_test.shape, Y_test.shape)
print ("Stage4 : Completed (probabilistic vecs extraction)")
print
#########################################################################################################################################

num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = 10

n=X_train.shape[0]
X_train1 = X_train[:, :n/2].astype('float32')
X_train2 = X_train[:, n/2:].astype('float32') 
X_train = np.hstack((X_train1, X_train2))

X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_test)

Y_train=Y_train
Y_test=Y_test

hkl.dump( X_train, 'X_CSIQ_train.hkl' )
hkl.dump( X_test, 'X_CSIQ_test.hkl' )
hkl.dump( Y_train, 'Y_CSIQ_train.hkl' )
hkl.dump( Y_test, 'Y_CSIQ_test.hkl' )

hkl.dump( Y_traintarget, 'Y_CSIQ_traintarget.hkl' )
hkl.dump( Y_testtarget, 'Y_CSIQ_testtarget.hkl' )

