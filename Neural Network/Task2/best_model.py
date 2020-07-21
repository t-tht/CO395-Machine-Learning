from param import round_1, round_2
import gc
import sys
from csv_stuff import sort_csv
from data_stuff import magic_data
from keras import backend as K

p={
    'hidden_layers':(2,5,1,1),
    'l1':(10,201,1,1),
    'l2':(10,201,1,1),
    'l3':(10,201,1,1),
    'l4':(10,201,1,1),
    'ac1':['relu','sigmoid','tanh'],
    'ac2':['relu','sigmoid','tanh'],
    'ac3':['relu','sigmoid','tanh'],
    'ac4':['relu','sigmoid','tanh'],
    'd1':(1,6,1,10),
    'd2':(1,6,1,10),
    'd3':(1,6,1,10),
    'd4':(1,6,1,10),
    'kernel_init':['random_normal', 'random_uniform'],
    'loss_func':['mse'],
    'metrics':['mae'],
    'val_split':[0.2],
    'epoch_size':[300],
    'batch_size':(10,101,5,1),
    'lr':(1,1001,10,1000)
}

mode = int(sys.argv[1])
round2_i = int(sys.argv[2])

x_train, y_train, x_test, y_test, _, _ = magic_data()

if mode == 1:
    # print('round 1')
    round_1(x_train, y_train, p)

if mode == 2:
    # print('round 2')

    round_2(x_train,y_train,x_test,y_test,round2_i)
    gc.collect()
    K.clear_session()


if mode == 3:
    print('==================== sorting round_1 ====================')
    sort_csv('result/round_1.csv')
if mode == 4:
    print('==================== sorting round_2 ====================')
    sort_csv('result/round_2.csv')
