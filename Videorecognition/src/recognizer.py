import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from labels import dictionary
from constants import NUMBER_OF_LETTERS, NUMBER_OF_PIXELS

class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.99):
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True

class NeuralNetwork():

    training_data = pd.read_csv('../../MNIST/sign_mnist_train.csv')
    testing_data = pd.read_csv('../../MNIST/sign_mnist_test.csv')
    
    def __init__(self):
        self.x_train = self.data_resizing(np.array([np.reshape(i, (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)) for i in self.training_data.iloc[:, 1:].values])) 
        self.y_train = self.data_labeling(self.training_data.iloc[:, 0])
        self.x_test = self.data_resizing(np.array([np.reshape(i, (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)) for i in self.testing_data.iloc[:, 1:].values])) 
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
        x = x.reshape(x.shape[0], NUMBER_OF_PIXELS, NUMBER_OF_PIXELS, 1)
        return x

    def make_model(self):

        callbacks = myCallback()   
        self.model = Sequential([
            Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(NUMBER_OF_PIXELS, NUMBER_OF_PIXELS, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation = 'relu'),
            Dropout(0.4),
            Dense(NUMBER_OF_LETTERS, activation='softmax')
        ])  

        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.model.fit(self.x_train, self.y_train,  epochs=10, callbacks=[callbacks])  
  
    def predict_pic(self, image):
        accuracy = self.model.predict(image).reshape(NUMBER_OF_LETTERS)
        pred = accuracy.round().astype(int)
        for i in range(0, len(pred)):
            if pred[i] == 1:
                return dictionary[i], accuracy[i]
        return 'I couldnt recognize ', 0        

