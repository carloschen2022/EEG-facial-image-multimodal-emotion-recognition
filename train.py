import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input

#EEG data
EEG = sio.loadmat("EEG_train.mat")
x= EEG['naodian'].reshape(9600,32,1)

l1=np.ones(3200)*2
l2=np.ones(3200)
l3=np.zeros(3200)
y=np.hstack((l1,l2,l3))
y = to_categorical(y)
#split EEG dataset
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=1234)

#Facial image data
imx=np.load('Facial image_train.npy')
imx=imx/255
l1=np.ones(3200,)*2
l2=np.ones(3200,)
l3=np.zeros(3200,)
l=np.hstack((l1,l2,l3))
imy=to_categorical(l)
#split facial image dataset
imx_train,imx_val,imy_train,imy_val = train_test_split(imx,imy,test_size=0.2,random_state=1234)

image_shape=Input(shape=(85,70,3))
eeg_shape=Input(shape=(32,1))

conv21 = Conv2D(32,kernel_size=3, activation='relu')(image_shape)
pool21 = MaxPooling2D(pool_size=(2, 2),padding="same")(conv21)
conv21b_b = BatchNormalization()(pool21)
conv22 = Conv2D(32, kernel_size=3, activation='relu')(conv21b_b)
pool22 =MaxPooling2D(pool_size=(2, 2),padding="same")(conv22)
conv23 = Conv2D(64, kernel_size=3, activation='relu')(pool22)
pool23 =MaxPooling2D(pool_size=(2, 2),padding="same")(conv23)
conv24 = Conv2D(64, kernel_size=3, activation='relu')(pool23)
conv25 = Conv2D(32, kernel_size=1, activation='relu')(conv24)
pool25 =MaxPooling2D(pool_size=(2, 2),padding="same")(conv25)
flat1=Flatten()(pool25)

Conv21 = Conv1D(32,kernel_size=3, activation='relu')(eeg_shape)
Pool21 = MaxPooling1D(pool_size=2,padding="same")(Conv21)
Conv21b_b = BatchNormalization()(Pool21)
Conv22 = Conv1D(32, kernel_size=3, activation='relu')(Conv21b_b)
Pool22 =MaxPooling1D(pool_size=2,padding="same")(Conv22)
flat2 = Flatten()(Pool22)

merge=Concatenate()([flat1,flat2])
flatten = Flatten()(merge)
hidden1 = Dense(128, activation='relu') (merge)
hidden2 = Dense(16,activation="relu")(hidden1)
output=Dense(3,activation="softmax")(hidden2)
model=Model(inputs=(image_shape,eeg_shape),outputs=output)

model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['CategoricalAccuracy'])
model.summary()
results=model.fit([imx_train,x_train],[imy_train,y_train],batch_size=64,epochs=30,validation_data=([imx_val,x_val],[imy_val,y_val]))
model.save("model_fusion.h5")

plt.plot(results.history['categorical_accuracy'],label="accuracy")
plt.plot(results.history['val_categorical_accuracy'],label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(results.history['loss'],label="loss")
plt.plot(results.history['val_loss'],label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


