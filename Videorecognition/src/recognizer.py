import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from labels import labels

NUMBER_OF_LETTERS = 24

class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.98):
        print("\nReached 98% accuracy so cancelling training!")
        self.model.stop_training = True

class NeuralNetwork():

    training_data = pd.read_csv('../../MNIST/sign_mnist_train.csv')
    testing_data = pd.read_csv('../../MNIST/sign_mnist_test.csv')
    

    def __init__(self):

        self.x_train = self.data_resizing(np.array([np.reshape(i, (28, 28)) for i in self.training_data.iloc[:, 1:].values])) 
        self.y_train = self.data_labeling(self.training_data.iloc[:, 0])
        self.x_test = self.data_resizing(np.array([np.reshape(i, (28, 28)) for i in self.testing_data.iloc[:, 1:].values])) 
        self.y_test = self.data_labeling(self.testing_data.iloc[:, 0])
        
        if  os.path.isfile('../saved_model/saved_model.pb'):
            print('Loading model')     
            self.model = load_model('../saved_model')
        
        else:    
            print('Working on new model')
            self.make_model()
            self.model.save('../saved_model') 

    def data_labeling(self, y):
        label_binrizer = LabelBinarizer()
        labels = label_binrizer.fit_transform(y)

        return labels

    def data_resizing(self, x):
        x = x / 255.0
        x = x.reshape(x.shape[0], 28, 28, 1)
        return x

    def feature_scaling(self, data):
        sc = StandardScaler()
        data = sc.fit_transform(data)

        return data

    def make_model(self):

        callbacks = myCallback() 
    
        self.model = Sequential([
            Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation = 'relu'),
            Dropout(0.20),
            Dense(NUMBER_OF_LETTERS, activation='softmax')
        ])  

        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.model.fit(self.x_train, self.y_train,  epochs=10, callbacks=[callbacks])  
  
    def predict_pic(self, image):
        pred = self.model.predict(image).round().astype(int).reshape(NUMBER_OF_LETTERS)
        for i in range(0, len(pred)):
            if pred[i] == 1:
                return labels[i]

