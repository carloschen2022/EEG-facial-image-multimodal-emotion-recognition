#Import related modules
import numpy as np
import tensorflow as tf
import scipy.io as sio
import  matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

#read test data
imx=np.load('test data/img_test.npy')
x = sio.loadmat("test data/EEG_test.mat")

#load model
model_EEG=load_model("prtrained model/model_EEG.h5")
model_img=load_model("prtrained model/model_img.h5")
model_fusion=load_model("prtrained model/model_fusion.h5")

#data:0-49（happiness）,50-99(calmness),100-149(sadness)
#Randomly select data for prediction

imx_predict=imx[60,:,:,:]
x_predict=x["naodian"][60,:,:]
img_predict=model_img.predict(imx_predict.reshape(1,85,70,3))
EEG_predict=model_EEG.predict(x_predict.reshape(1,32,1))
fusion_predict=model_fusion.predict([imx_predict.reshape(1,85,70,3),x_predict.reshape(1,32,1)])

img_index=np.argmax(img_predict)
EEG_index=np.argmax(EEG_predict)
fusion_index=np.argmax(fusion_predict)

emotion_classes=["sadness","calmness","happyness"]
print("The category predicted by the  single image model is:",emotion_classes[img_index])
print("The category predicted by the  single EEG model is:",emotion_classes[EEG_index])
print("The category predicted by the  cross-media fusion model is:",emotion_classes[fusion_index])