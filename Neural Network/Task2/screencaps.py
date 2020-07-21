import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.activations import relu, sigmoid, tanh
from keras import backend as K
from keras.callbacks import EarlyStopping
from data_stuff import magic_data

x_train, y_train, x_test, y_test,_,_ = magic_data()


model = Sequential()

#   input layer + hidden layer 1
model.add(Dense(10,input_dim=3,activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))

#   hidden layer 2
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))

#   hidden layer 3
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))

#   output layer
model.add(Dense(3,activation='linear'))

#   compiles the model, but doesn't train the model
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss=['mse'],
                metrics=['mae'])
#   trains the model
history = model.fit(x_train, y_train,
                validation_split = 0.2,
                epochs = 300,
                batch_size = 50,
                shuffle=True,
                verbose=2)

train_result = model.evaluate(x_test, y_test,verbose=0)

print(train_result)
