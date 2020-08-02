import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from skimage import io
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import NUMBER_OF_LETTERS, NUMBER_OF_PIXELS, DICTIONARY

class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.99):
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True

class NeuralNetwork():

    training_dir = '../../MNIST/train'
    testing_dir = '../../MNIST/test'
    
    def __init__(self):       
        if  os.path.isfile('../saved_model/saved_model.pb'):
            print('Loading model')     
            self.model = load_model('../saved_model')
        
        else:    
            print('Working on new model')
            self.make_model()
            self.model.save('../saved_model') 

    def make_model(self):

        callbacks = myCallback()   
        self.model = Sequential([
            Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(NUMBER_OF_PIXELS, NUMBER_OF_PIXELS, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.4),
            Dense(128, activation = 'relu'),
            Dense(NUMBER_OF_LETTERS, activation='softmax')
        ])  

        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        # self.model.fit(self.x_train, self.y_train,  epochs=10, callbacks=[callbacks])  
        self.fit_model()

    def fit_model(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        train_generator = train_datagen.flow_from_directory(
            self.training_dir,
            batch_size=10,
            class_mode='categorical',
            target_size=(NUMBER_OF_PIXELS, NUMBER_OF_PIXELS),
            color_mode = 'grayscale'
        )     
        validation_datagen  = ImageDataGenerator( rescale = 1.0/255. )
        validation_generator = validation_datagen.flow_from_directory(
            self.training_dir,
            batch_size=10,
            class_mode  = 'categorical',
            target_size = (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS),
            color_mode = 'grayscale'
        )
        self.history = self.model.fit(train_generator,epochs=60,verbose=1,validation_data=validation_generator)
  
    def predict_pic(self, image):
        accuracy = self.model.predict(image).reshape(NUMBER_OF_LETTERS)
        pred = accuracy.round().astype(int)
        for i in range(0, len(pred)):
            if pred[i] == 1:
                return DICTIONARY[i], accuracy[i]
        return 'I couldnt recognize ', 0        

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()



