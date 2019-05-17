
#Note: We have used the matlab codes of already existing NSS feature extractors by Alan Bovik. 
#Note: Use SVR trained model with 80 features

import cv2
import numpy as np
from keras.models import model_from_json
from scipy.stats import mode


image_Path='img1.bmp'
cnn_Model_Path='./TrainedModels/Live_Challenge/3SCNN_Live_Chl.json'
cnn_Model_Weights_Path='./TrainedModels/Live_Challenge/3SCNN_Weights_Live_Chl.h5'
svr_Model_Path ='./TrainedModels/Live_Challenge/SVR_80features.joblib'

img=cv2.resize(cv2.imread(image_Path),(500,500))
#print np.shape(img)


json_file = open(cnn_Model_Path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(cnn_Model_Weights_Path)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy




window_shape = (224, 224, 3)
step=38


from skimage.util import view_as_windows
B = view_as_windows(img, window_shape,step=step)
patch=B.reshape(B.shape[0]*B.shape[1],224,224,3)
patch = patch.astype('float32')
print np.max(patch)
patch /= np.max(patch)
probi=model.predict(patch)
he=mode(np.argmax(probi, axis=1))[0][0]
ti=np.array([[0]*10])
ti[0][he]=1

#print ti


import matlab.engine
eng = matlab.engine.start_matlab()


NSSFeatures = np.array(eng.dataSetFeat(image_Path)) #For a Single image
#print NSSFeatures


from sklearn.externals import joblib
#joblib.dump(pca, 'pca.joblib')
pca = joblib.load('pca.joblib')
#NSSFeatures = pca.fit_transform(np.reshape(NSSFeatures,(1169,234)))
NSSFeatures = pca.transform(NSSFeatures)
#print (np.shape(NSSFeatures))


SVM_Features=np.concatenate((NSSFeatures,ti),axis=1)

#print (np.shape(SVM_Features))

clf2 = joblib.load(svr_Model_Path) 

score=clf2.predict(SVM_Features)

print score


