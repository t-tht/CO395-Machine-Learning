import numpy as np
import json
import random
import gc

from keras.models import load_model
from keras import backend as K
from model import fm_model
from data_stuff import feature_scaling, normalize_output, split_data
from csv_stuff import get_permutation_csv, write_csv


def round_1(x_train, y_train, param, write = 'result/round_1.csv'):
    rand_p = rand_params(param)

    history, model = fm_model(x_train, y_train, rand_p)
    train_result = model.evaluate(x_train, y_train,verbose=0)
    validation_result = [history.history['val_loss'][-1] ,history.history['val_mean_absolute_error'][-1]]

    jsonparam = json.dumps(rand_p)
    row = [train_result[0], train_result[1], validation_result[0], validation_result[1], jsonparam]
    write_csv(write, row)

def round_2(x_train, y_train, x_test, y_test, param_no, fold=10, read = 'result/round_1.csv', write = 'result/round_2.csv'):
    minp={
        'hidden_layers':1,
        'l1':1,
        'l2':1,
        'l3':1,
        'l4':1,
        'l5':1,
        'ac1':'relu',
        'ac2':'relu',
        'ac3':'relu',
        'ac4':'relu',
        'ac5':'relu',
        'd1':0,
        'd2':0,
        'd3':0,
        'd4':0,
        'd5':0,
        'kernel_init':'random_uniform',
        'loss_func':'mse',
        'metrics':'mae',
        'val_split':0.2,
        'epoch_size':2,
        'batch_size':1000,
        'lr':1
    }
    permutation = get_permutation_csv(read)
    param = permutation[param_no]['param']
    vloss = []
    vmetric = []
    loss = []
    metric = []
    highscore = 999
    _,highmodel = fm_model(x_train, y_train, minp)

    for i in range(fold):
        print('==================== round_2: validating {} ===================='.format(i+1))
        history, model = fm_model(x_train, y_train, param)
        train_result = model.evaluate(x_train, y_train,verbose=0)

        loss.append(train_result[0])
        metric.append(train_result[1])
        vloss.append(history.history['val_loss'][-1])
        vmetric.append(history.history['val_mean_absolute_error'][-1])

        if vmetric[-1] < highscore:
            highscore = vmetric[-1]
            high_model = model

    vloss_avg = sum(vloss)/len(vloss)
    vmetric_avg = sum(vmetric)/len(vmetric)
    loss_avg = sum(loss)/len(loss)
    metric_avg = sum(metric)/len(metric)

    jsonparam = json.dumps(param)
    savedir = 'result/model_{}.h5'.format(param_no)
    highmodel.save(savedir)
    # jsonmodel = highmodel.to_json()
    # jsonweight = highmodel.get_weights()
    test_score = model.evaluate(x_test, y_test)
    row = [loss_avg, metric_avg, vloss_avg, vmetric_avg ,jsonparam, test_score[1]]
    write_csv(write, row)

    gc.collect()
    del model
    K.clear_session()


def rand_params(p):
    rand_p = p
    for items in rand_p:
        next = rand_p[items]
        if isinstance(next, tuple):
            if items not in ['d1', 'd2', 'd3', 'd4', 'd5', 'lr']:
                next = int(random.randrange(next[0], next[1] ,next[2])/next[3])
            else:
                next = random.randrange(next[0], next[1] ,next[2])/next[3]
        if isinstance(next, list):
            next = random.choice(next)
        rand_p[items] = next
    return rand_p
