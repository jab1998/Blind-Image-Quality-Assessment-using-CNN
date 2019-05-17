_____________________________________________________________________________________
Blind Image Quality Assessment using a Combination of Statistical Features and CNN
_______________________________________________________________________________________


This folder consists of four sub folders.
1. Score_Prediction
2. LiveChallenge_Files
3. IQA_Files
4. CSIQ_Files

1. Score_Prediction:This folder contains NSS feature set extraction files and a Quality predictor which predics the quality of an image.
	In 'Image_Quality_Predictor.py', if we give an image path then it will return its predicted Quality Score.
	In 'Applying_PCA.py', the dimensionality gets reduced to 72 (from 234)

2. LiveChallenge_Files: This folder contains all the codes of CNN and SVR training and testing for LiveChallenge dataset.

3. LiveIQA_Files: This folder contains all the codes of CNN and SVR training and testing for LiveIQA dataset.

4. CSIQ_Files: This folder contains all the codes of CNN and SVR training and testing for CSIQ dataset.

You can find the trained models and extracted NSS features in the link provided below:

https://drive.google.com/drive/folders/1Ytw0pqASrSlvvrbEJggivmtAJwMl2qUO?usp=sharing



After downloading the models run the 'Image_Quality_Predictor.py' after updating the following variables with the trained model links:

image_Path='img1.bmp'

cnn_Model_Path='./TrainedModels/Live_Challenge/3SCNN_Live_Chl.json'

cnn_Model_Weights_Path='./TrainedModels/Live_Challenge/3SCNN_Weights_Live_Chl.h5'

svr_Model_Path ='./TrainedModels/Live_Challenge/SVR_70features.joblib'

________________________________________________________________________________________
