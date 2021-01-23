import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import to_categorical


#DATA COLLECTION AND CATEGORIES DEFINITION
Datadir = "data"
Categories = ["Benign130" , "Non-cancer130" , "Malignant130"]

#print(img_array)
#print(img_array.shape)

#DATA PRE-PROCESSING
img_size = 150


training_data = []
def create_training_data():
    for category in Categories:
        path = os.path.join(Datadir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
import random
random.shuffle(training_data)

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,img_size,img_size,1)
Y = np.array(Y)

Y= to_categorical(Y)

#MODELS BUILDING

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

X = X/255.0

batch_size = [32, 64, 128]
epochs = [10, 15, 20, 50]
dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]


#------------------------------------------------------------------------------------------------------------

for dense_layer in dense_layers:       #LOOK FOR THE BEST COMBINATION
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME4 = "{}-conv-{}-nodes-{}dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time( )))
            tensorboard = TensorBoard(log_dir="logsmulti/{}".format(NAME4))
            print(NAME4)

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))
            model.add(Flatten())


            model.add(Dense(3))
            model.add(Activation("softmax"))

            model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics= ["accuracy"])
            model.fit(X, Y, batch_size=32, epochs=20, validation_split=0.2, callbacks= [tensorboard])
#------------------------------------------------------------------------------------------------------------------------
#FIT THE MODEL WITH THE BEST VALUES OBTAINED

dense_layers = [1]
layer_sizes = [64]
conv_layers = [2]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME4 = "{}-conv-{}-nodes-{}dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time( )))
            tensorboard = TensorBoard(log_dir="logsmultib/{}".format(NAME4))


            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1): #we had one before

                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(dense_layer):  #First dense must have a flatten layer
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))   #helps overfitting

            model.add(Flatten())#aiuta il non overfitting


            model.add(Dense(3))
            model.add(Activation("softmax"))

            model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics= ["accuracy"])
            for batch in batch_size:
                for epoch in epochs:
                    #model.fit(X, Y, batch_size=batch, epochs=epoch, validation_split=0.2)  #Hypertuning of epochs and batch_size
                    model.fit(X, Y, batch_size=32, epochs=15, validation_split=0.2, callbacks = [tensorboard])




model.save("Cancer_prediction")
#82-86%  accuracy con batch 32 e 15 epoche , come previsto
# val_loss 0.73

model2 = tf.keras.models.load_model("Cancer_prediction")
print(model2.summary())

# 2 convolutional layers , with 64 filters 3x3
