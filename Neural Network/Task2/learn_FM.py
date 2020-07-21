import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pylab as plt
from keras.layers import Activation, Dense, LeakyReLU
from keras.activations import relu, sigmoid, tanh
from keras.models import load_model
from keras.models import Sequential
# from keras import backend as K

from csv_stuff import get_permutation_csv
from data_stuff import magic_data, normalize_output, scale_input
from model import fm_model
from load_model import load_model


def predict_hidden(hidden_data):
    #input = test dataset input
    #output = predicted output based on input, model object, data during training,
    #           and best parameters

    #   get features from training set
    _, _, _,_,train_feat,_ = magic_data()

    #   pre process data
    #   need training data features for scaling input
    x_test = hidden_data[:,:3]
    y_true = hidden_data[:,3:]
    x_test = scale_input(x_test, train_feat)
    #   get best parameters
    permu = get_permutation_csv('result/final.csv')
    param = permu[0]['param']

    print("==================== please wait ====================")
    # make base model with param
    model = load_model(param)
    # load weights into it
    model.load_weights('weights.h5')
    model._make_train_function()

    y_pred = model.predict(x_test)
    return y_pred, y_true, model, param, train_feat




def evaluate_architecture(hidden_data, y_pred, model, param, train_feat):

    #   y_pred : predicted output from preddict_hidden()
    #   model : get from preddict_hidden()
    #   param : best param output from preddict_hidden()
    #   train_feat : needed for scaling

    x_test = hidden_data[:,:3]
    y_true = hidden_data[:,3:]
    x_test = scale_input(x_test, train_feat)

    y_pred = model.predict(x_test)
    mae = np.mean(np.abs(y_true - y_pred), axis = 0)

    print('==================== Layers information of the model ====================')
    model.summary()
    print("==================== optimal parameters ====================")

    for item in param:
        print("{} : {}".format(item, param[item]))

    print("==================== evaluation ====================")
    print("mae for output x, y, and z: {}".format(mae))
    print("total mae : {}".format(sum(mae)/len(mae)))





#   for testing
_, _, x_test,y_test,_,_ = magic_data()

hidden = np.concatenate((x_test, y_test), axis=1)

y_pred, y_true, model, param, train_feat = predict_hidden(hidden)
evaluate_architecture(hidden, y_pred, model, param, train_feat)
