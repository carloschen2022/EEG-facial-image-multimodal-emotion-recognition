#Import related modules
import numpy as np
import tensorflow as tf
import scipy.io as sio
import  matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

#Read the EEG test set
naodian = sio.loadmat("test data/EEG data.mat")
x= naodian['naodian'].reshape(3840,32,1)

#0 represents sadness,1 represents calmness, and 2 represents happyness
l1=np.ones(1280)*2
l2=np.ones(1280)
l3=np.zeros(1280)
y=np.hstack((l1,l2,l3))
y = to_categorical(y)

#Read facial image test set
imx=np.load('test data/Facial image data.npy')
imx=imx/255

l1=np.ones(1280,)*2
l2=np.ones(1280,)
l3=np.zeros(1280,)
l=np.hstack((l1,l2,l3))
imy=to_categorical(l)

#load model
model_fusion=load_model("prtrained model/model_fusion.h5")

#model test
test_accuracy=model_fusion.evaluate([imx,x],[imy,y])
print("The accuray of the test model is:",test_accuracy[1])

#data:0-1280（happiness）,1281-2560(calmness),2561-3780(sadness)
#Randomly select data for prediction
imx_predict=imx[300,:,:]
x_predict=x[300,:,:]
y_predict=model_fusion.predict([imx_predict.reshape(1,85,70,3),x_predict.reshape(1,32,1)])
class_index=np.argmax(y_predict)
emotion_classes=["sadness","calmness","happyness"]
print("The category predicted by the model is:",emotion_classes[class_index])
