from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras
import numpy as np

def mnisty():
 (x_r, y_r),(x_e,y_e) = mnist.load_data()
 x_r = x_r.reshape(60000,1,28,28)/256.0
 x_e = x_e.reshape(10000,1,28,28)/256.0
 y_r = keras.utils.to_categorical(y_r, num_classes = 10)
 y_e = keras.utils.to_categorical(y_e, num_classes = 10)
 return (x_r, y_r), (x_e, y_e)


