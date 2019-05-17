import pandas as pd
import glob
import numpy as np
import cv2
import hickle as hkl 
import matplotlib.pyplot as plt
DataSet='/home/jbaravind/ChallengeDB_release/ChallengeDB_release/CSIQ_Files/CSIQ/'
filenames = glob.glob(DataSet+"dst_imgs/awgn/*.png")
filenames.sort()

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
from scipy.stats import spearmanr


NSSfeatures = hkl.load( 'NSSEnrichedfeatures_LiveIQA2.hkl' )
#print (np.shape(NSSfeatures))

#Live Challenge 1169
#CSIQ 866
#IQA 982

from sklearn.decomposition import PCA
NSSfeatures=np.reshape(NSSfeatures,(982,234))
'''
pca = PCA().fit(NSSfeatures)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Principal Components Vs Variance')
plt.show()

print np.cumsum(pca.explained_variance_ratio_)
'''
print (np.shape(NSSfeatures))
pca = PCA(n_components=90)
NSSfeatures = pca.fit_transform(np.reshape(NSSfeatures,(982,234)))
print (np.shape(NSSfeatures))


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
import hickle as hkl
from tqdm import tqdm
from scipy.stats import spearmanr

Labels=hkl.load('Live_IQA_Labels.hkl')

#Image_Names=si.loadmat('./Data/AllImages_release.mat')['AllImages_release']
#print np.array([cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])).shape for i in range(1169)])

Images= hkl.load('Live_IQA_Images.hkl')


print ("Stage:1 Read all the Images")

json_file = open('./Models/model_LIVE_IQA_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("./Models/model_LIVE_IQA_final.h5")

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

One_hot=t



SVM_Features=np.concatenate((NSSfeatures,One_hot),axis=1)

print (np.shape(SVM_Features))

X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(SVM_Features, Labels, test_size=0.15, random_state=42)

print X_train.shape,Y_traintarget.shape

print X_test.shape, Y_testtarget.shape


from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
'''
e=[0.01*(i+1) for i in range(0,100,20)]
c=[0.5*(i+1)   for i in range(0,5)] 
g=[0.1*(10**-i) for i in range(2,5)]
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':c,'gamma': g,'epsilon':e}
svr = SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, Y_traintarget)
print clf.best_params_
'''

from sklearn.externals import joblib
#joblib.dump(clf, 'SVR_ver3.joblib')

clf2 = joblib.load('SVR_ver4.joblib') 

k=clf2.predict(X_test)
print k[0:50]
print Y_testtarget[0:50]
print spearmanr(Y_testtarget,k)
