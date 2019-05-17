import pandas as pd
import glob
import numpy as np
import cv2
import hickle as hkl 
import matplotlib.pyplot as plt


NSSfeatures = hkl.load( 'NSSEnrichedfeatures_LiveChallenge.hkl' ) #.hkl file contains the extracted features from 'FEATURE_EXTRACTOR.py' 



from sklearn.decomposition import PCA
NSSfeatures=np.reshape(NSSfeatures,(1169,234))
print np.shape(NSSfeatures)
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
NSSfeatures = pca.fit_transform(np.reshape(NSSfeatures,(1169,234)))
print (np.shape(NSSfeatures))

from sklearn.externals import joblib
joblib.dump(pca, 'pca.joblib')
pca = joblib.load('pca.joblib')
#NSSfeatures = pca.fit_transform(np.reshape(NSSfeatures,(1169,234)))
#print (np.shape(NSSfeatures))

