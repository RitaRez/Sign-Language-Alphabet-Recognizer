import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def data_preprocessing(y):
    new_y = []
    
    for el in y:
        arr = [0] * 26
        arr[el - 1] = 1
        new_y.append(arr)
    
    return new_y

def feature_scaling(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)

    return data

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

def neural_network(x_train, y_train):
    
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=26, activation='sigmoid')
    ])

    nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    nn.fit(x_train, y_train,  epochs=3)  

    return nn
    
def main():
    data = pd.read_csv('./MNIST/sign_mnist_train.csv')
    x = data.iloc[:, 1:-1]
    y = data_preprocessing(data.iloc[:, 0])

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)

    x_train = feature_scaling(x_train)
    x_test = feature_scaling(x_test)
    
    nn = neural_network(x_train, y_train)

