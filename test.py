#Import related modules
import numpy as np
import tensorflow as tf
import scipy.io as sio
import  matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

#Read the EEG test set
naodian = sio.loadmat("EEG data.mat")
x= naodian['naodian'].reshape(3840,32,1)

#0 represents sadness,1 represents calmness, and 2 represents happyness
l1=np.ones(1280)*2
l2=np.ones(1280)
l3=np.zeros(1280)
y=np.hstack((l1,l2,l3))
y = to_categorical(y)

#Read facial image test set
imx=np.load('Facial image data.npy')
imx=imx/255

l1=np.ones(1280,)*2
l2=np.ones(1280,)
l3=np.zeros(1280,)
l=np.hstack((l1,l2,l3))
imy=to_categorical(l)

#load model
model_EEG=load_model("prtrained model/model_EEG.h5")
model_img=load_model("prtrained model/model_img.h5")
model_fusion=load_model("prtrained model/model_fusion.h5")

#model test
img_test=model_img.evaluate(imx,imy)
EEG_test=model_EEG.evaluate(x,y)
fusion_test=model_fusion.evaluate([imx,x],[imy,y])
print("The test accuracy of the single image model is:",img_test[1])
print("The test accuracy of the single EEG model is:",EEG_test[1])
print("The test accuracy of the cross-media fusion model is:",fusion_test[1])

#data:0-1280（happiness）,1281-2560(calmness),2561-3780(sadness)
#Randomly select data for prediction
imx_predict=imx[1000,:,:]
x_predict=x[1000,:,:]
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
