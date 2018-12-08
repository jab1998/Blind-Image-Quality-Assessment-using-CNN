#########################################################################################################################################
import scipy.io as si
import numpy as np
import cv2
import os
import func as f
import ggd
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from scipy import signal
import math
from keras.models import model_from_json
from sklearn.svm import SVR
import pickle

Labels=si.loadmat('./Data/AllMOS_release.mat')['AllMOS_release'][0]

Image_Names=si.loadmat('./Data/AllImages_release.mat')['AllImages_release']

Images= np.array([cv2.resize(cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])),(100,100)) for i in range(1169)])

print "Stage 1 Completed"

#########################################################################################################################################

print len(Labels),Images.shape


i=0
for a in range(len(Labels)):
	if i==0:
		patch=image.extract_patches_2d(Images[a], (64,64),0.0009)
		patchlabel=np.array(patch.shape[0]*[Labels[a]])
		i=1
	else:
		patchi=image.extract_patches_2d(Images[a], (64,64),0.0009)
		patchlabeli=np.array(patchi.shape[0]*[Labels[a]])
		patch=np.append(patch,patchi,axis=0)
		patchlabel=np.append(patchlabel,patchlabeli,axis=0)	

print (patch.shape)
print (patchlabel.shape)	

print ("Stage2: Completed (Patch Extraction)")

X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(patch, patchlabel, test_size=0.1, random_state=42)

print X_train.shape,Y_traintarget.shape

print X_test.shape, Y_testtarget.shape

window = signal.gaussian(49, std=7)
window=window/sum(window)
NSStrainfeatures=[]
NSStestfeatures=[]
count=0

for I in X_train:
	print count
	count=count+1 
        #I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
        I = I.astype(np.float32)
        u = cv2.filter2D(I,-1,window)
        u = u.astype(np.float32)
	
        diff=pow(I-u,2)
        diff = diff.astype(np.float32)
        sigmasquare=cv2.filter2D(diff,-1,window)
        sigma=pow(sigmasquare,0.5)

        Icap=(I-u)/(sigma+1)
        Icap = Icap.astype(np.float32)
        gamparam,sigma = ggd.estimateggd(Icap)
        feat=[gamparam,sigma]



        shifts = [ (0,1), (1,0) , (1,1) , (-1,1)];
        for shift in shifts:
                shifted_Icap= np.roll(Icap,shift,axis=(0,1))
                pair=Icap*shifted_Icap
                alpha,leftstd,rightstd=ggd.estimateaggd(pair)
                const=(np.sqrt(math.gamma(1/alpha))/np.sqrt(math.gamma(3/alpha)))
                meanparam=(rightstd-leftstd)*(math.gamma(2/alpha)/math.gamma(1/alpha))*const;
                feat=feat+[alpha,leftstd,rightstd,meanparam]

        feat=np.array(feat)
        NSStrainfeatures.append(feat)

print NSStrainfeatures

for I in X_test:
        #I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
        I = I.astype(np.float32)
        u = cv2.filter2D(I,-1,window)
        u = u.astype(np.float32)

        diff=pow(I-u,2)
        diff = diff.astype(np.float32)
        sigmasquare=cv2.filter2D(diff,-1,window)
        sigma=pow(sigmasquare,0.5)

        Icap=(I-u)/(sigma+1)
        Icap = Icap.astype(np.float32)
        gamparam,sigma = ggd.estimateggd(Icap)
        feat=[gamparam,sigma]



        shifts = [ (0,1), (1,0) , (1,1) , (-1,1)];
        for shift in shifts:
                shifted_Icap= np.roll(Icap,shift,axis=(0,1))
                pair=Icap*shifted_Icap
                alpha,leftstd,rightstd=ggd.estimateaggd(pair)
                const=(np.sqrt(math.gamma(1/alpha))/np.sqrt(math.gamma(3/alpha)))
                meanparam=(rightstd-leftstd)*(math.gamma(2/alpha)/math.gamma(1/alpha))*const;
                feat=feat+[alpha,leftstd,rightstd,meanparam]

        feat=np.array(feat)
        NSStestfeatures.append(feat)

print NSStestfeatures

print np.max(NSStestfeatures)
print np.min(NSStestfeatures)
print "NSS Feature Extraction Completed"

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 

loaded_model.compile(loss='KLD', optimizer='adam', metrics=['mae'])
Train_Prob_vec=loaded_model.predict(X_train)
Test_Prob_vec=loaded_model.predict(X_test)

SVM_Train_Features=np.concatenate((NSStrainfeatures,Train_Prob_vec),axis=1)
SVM_Test_Features=np.concatenate((NSStestfeatures,Test_Prob_vec),axis=1)

print (SVM_Train_Features.shape)
print (SVM_Test_Features.shape)

print ("SVM Training")
########################################################################################################################################

clf = SVR(gamma='0.05', C=1.0, epsilon=0.02)
clf.fit(SVM_Train_Features, Y_traintarget)
###np.concatenate((a,b),axis=1).shape


from sklearn.externals import joblib
joblib.dump(clf, 'SVR.joblib')

clf2 = joblib.load('SVR.joblib') 

clf2.predict(SVM_Test_Features)


