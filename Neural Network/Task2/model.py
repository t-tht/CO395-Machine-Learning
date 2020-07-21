import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.activations import relu, sigmoid, tanh
from keras import backend as K
from keras.callbacks import EarlyStopping


def fm_model(x, y, p):
    model = Sequential()

    model.add(Dense(p['l1'],input_dim =3,kernel_initializer=p['kernel_init'], activation=p['ac1']))
    model.add(Dropout(p['d1']))

    for i in range(4):
        layerstr = "l"+str(i+2)
        dropoutstr = "d"+str(i+2)
        activationstr = "ac"+str(i+2)
        model.add(Dense(p[layerstr], activation=p[activationstr]))
        model.add(Dropout(p[dropoutstr]))

    model.add(Dense(3, activation='linear'))

    K.set_epsilon(1)

    model.compile(  optimizer=tf.train.AdamOptimizer(p['lr']),
                    loss=[p['loss_func']],
                    metrics=[p['metrics']])

    history = model.fit(x=x,y=y,
                        validation_split = 0.2,
                        epochs = p['epoch_size'],
                        batch_size = p['batch_size'],
                        shuffle=True,
                        verbose=0,
                        callbacks=[EarlyStopping(  monitor='val_mean_absolute_error',
                                                    min_delta=0.1,
                                                    patience=30,
                                                    verbose=0,
                                                    mode='auto'
                                                    )]
                        )

    return history, model
