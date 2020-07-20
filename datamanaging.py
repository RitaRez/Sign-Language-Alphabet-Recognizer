import pandas as pd 
import os
import numpy as np 

from Videorecognition.src.constants import NUMBER_OF_LETTERS, NUMBER_OF_PIXELS, DICTIONARY
from skimage import io

train_dir = './MNIST/train'
test_dir = './MNIST/test'

os.makedirs(train_dir)
os.makedirs(test_dir)

for key in DICTIONARY:
    path = os.path.join(train_dir, DICTIONARY[key])
    os.makedirs(path)
    path = os.path.join(test_dir, DICTIONARY[key])
    os.makedirs(path)

train_dataset = pd.read_csv('MNIST/sign_mnist_train.csv').iloc[:, :].values
test_dataset = pd.read_csv('MNIST/sign_mnist_test.csv').iloc[:, :].values

count = 0
for i in test_dataset:
    if(i[0] >= 9):
        i[0] -= 1
    image_path = os.path.join(test_dir, DICTIONARY[i[0]], str(count)+'.jpg')
    image = np.array(i[1:]).reshape(NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)
    io.imsave(image_path, image)
    count+= 1

for i in train_dataset:
    if(i[0] >= 9):
        i[0] -= 1
    image_path = os.path.join(train_dir, DICTIONARY[i[0]], str(count)+'.jpg')
    image = np.array(i[1:]).reshape(NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)
    io.imsave(image_path, image)
    count+= 1    