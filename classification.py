import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.activations import linear, relu, sigmoid


data  = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = data.load_data()

xtrain = tf.keras.utils.normalize(xtrain, axis = 1)
xtest = tf.keras.utils.normalize(xtest, axis = 1)


model = Sequential(
    [
        tf.keras.layers.Flatten(input_shape = (28, 28)),
        Dense(128, activation = 'relu', name = "L1"),
        Dense(128, activation = 'relu', name = "L2"),
        Dense(10, activation = 'linear', name = "L3")
    ], name = "my_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
) 


model.fit(xtrain, ytrain, epochs = 5)
model.save('handwrittenmodel.h5')


newmodel = keras.models.load_model('handwrittenmodel.h5')