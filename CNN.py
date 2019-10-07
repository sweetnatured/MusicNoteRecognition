import numpy as np
import matplotlib.pyplot as plt
import os
import  random
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

DATADIR = "/home/salih/Downloads/salih nota/img"
CATEGORİES_TRAİN=["do","la","mi","re","si","sol"]
IMG_SIZE=50


training_data=[]


def create_training_data():

  for category in CATEGORİES_TRAİN:

     path=os.path.join(DATADIR, category)
     class_number=CATEGORİES_TRAİN.index(category)

     for img in os.listdir(path):
       try:
         img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
         norm_image = cv2.normalize(new_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
         training_data.append([norm_image,class_number])
       except Exception as e:
           pass


DATADIR1 = "/home/salih/Downloads/salih nota/img/test"
CATEGORİES_TRAİN1=["do","la","mi","re","si","sol"]
IMG_SIZE=50

testing_data=[]
def create_testing_data():

    for category in CATEGORİES_TRAİN1:

        path = os.path.join(DATADIR1, category)
        class_number = CATEGORİES_TRAİN1.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                norm_image = cv2.normalize(new_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                           dtype=cv2.CV_32F)
                testing_data.append([norm_image, class_number])
            except Exception as e:
                pass




create_training_data()
random.shuffle(training_data)

create_testing_data()
random.shuffle(testing_data)

Features_train=[]
Labels_train=[]

for features,labels in training_data:

    Features_train.append(features)
    Labels_train.append(labels)

X = np.array(Features_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

Features_test=[]
Labels_test=[]
for features,labels in testing_data:

    Features_test.append(features)
    Labels_test.append(labels)

T = np.array(Features_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Labels_train, batch_size=32, epochs=3, validation_split=0.2, callbacks=[tensorboard])




liste=[]

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        liste.append(frame)
        M = np.array(liste).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        pr=model.predict(M)
        print(np.argmax(pr))
        print("{} written!".format(img_name))
        img_counter += 1
        liste.clear()

cam.release()

cv2.destroyAllWindows()