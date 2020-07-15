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


def data_labeling(y):
    label_binrizer = LabelBinarizer()
    labels = label_binrizer.fit_transform(y)

    return labels

def data_resizing(x):
    x = x / 255.0
    x = x.reshape(x.shape[0], 28, 28, 1)
    return x

def feature_scaling(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)

    return data

def neural_network(x_train, y_train):

    callbacks = myCallback() 
   
    nn = Sequential([
        Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation = 'relu'),
        Dropout(0.20),
        Dense(NUMBER_OF_LETTERS, activation='softmax')
    ])  

    nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    nn.fit(x_train, y_train,  epochs=10, callbacks=[callbacks])  

    return nn
    
def model_loader(x_train, y_train):
    if  os.path.isfile('../saved_model/saved_model.pb'):
        print('Loading model')     
        nn = load_model('saved_model')
    
    else:    
        print('Working on new model')
        nn = neural_network(x_train, y_train)
        nn.save('../saved_model') 

    return nn    

def predict_pic(image, nn):
    pred = nn.predict(image).round().astype(int).reshape(NUMBER_OF_LETTERS)
    for i in range(0, len(pred)):
        if pred[i] == 1:
            print(labels[i])
            return labels[i]

def main():
    training_data = pd.read_csv('../../MNIST/sign_mnist_train.csv')
    testing_data = pd.read_csv('../../MNIST/sign_mnist_test.csv')

    x_train = data_resizing(np.array([np.reshape(i, (28, 28)) for i in training_data.iloc[:, 1:].values])) 
    y_train = data_labeling(training_data.iloc[:, 0])
    
    x_test = data_resizing(np.array([np.reshape(i, (28, 28)) for i in testing_data.iloc[:, 1:].values])) 
    y_test = data_labeling(testing_data.iloc[:, 0])
    
    nn = model_loader(x_train, y_train)

    var = x_test[0].reshape(1, 28, 28, 1)
    
    pred = nn.predict(x_test).round().astype(int)
    print(pred)
    print(accuracy_score(pred, y_test))
    #predict_pic(var, nn)
    
main()