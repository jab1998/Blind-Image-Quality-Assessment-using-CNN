import pandas as pd
import glob
import numpy as np
import cv2
import hickle as hkl 
import matplotlib.pyplot as plt
#DataSet='/home/jbaravind/ChallengeDB_release/ChallengeDB_release/CSIQ_Files/CSIQ/'
#filenames = glob.glob(DataSet+"dst_imgs/awgn/*.png")
#filenames.sort()
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
from scipy.stats import spearmanr

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

NSSfeatures = hkl.load( 'NSSEnrichedfeatures_CSIQ2.hkl' )
#print (np.shape(NSSfeatures))

#Live Challenge 1169
#CSIQ 866
#IQA 

from sklearn.decomposition import PCA
NSSfeatures=np.reshape(NSSfeatures,(866,234))
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
NSSfeatures = pca.fit_transform(np.reshape(NSSfeatures,(866,234)))
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


'''
j=0
for Y in tqdm(Labels):
	One_hoti= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		One_hot=One_hoti
		j=1
	else:
		One_hot=np.append(One_hot,One_hoti,axis=0)

print (np.shape(One_hot))
'''
json_file = open('./Models/VGG_CSIQ.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("./Models/VGG_CSIQ.h5")

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

from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
'''
print "Grid search started"

e=[0.01*(i+1) for i in range(0,100,20)]
c=[0.5*(i+1)   for i in range(0,5)] 
g=[0.1*(10**-i) for i in range(2,5)]
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':c,'gamma': g,'epsilon':e}
svr = SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, Y_traintarget)
print clf.best_params_
'''
print "Completed"

from sklearn.externals import joblib
#joblib.dump(clf, 'SVR_ver4.joblib')

clf2 = joblib.load('SVR_ver3.joblib') 

k=clf2.predict(X_test)
print k[0:50]
print Y_testtarget[0:50]
print spearmanr(Y_testtarget,k)
