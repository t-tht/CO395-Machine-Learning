import numpy as np
import os
def magic_data():

    # train, test = split_data()
    # train_feat = []
    # test_feat = []

    train = np.loadtxt('data/train_origin.dat')
    test = np.loadtxt('data/test_origin.dat')


    train, train_feat = feature_scaling(train)
    test, test_feat = feature_scaling(test)
    _, train_feat = feature_scaling(train)
    _, test_feat = feature_scaling(test)

    x_train = train[:,:3]
    y_train = train[:,3:]

    x_test = test[:,:3]
    y_test = test[:,3:]

    return x_train, y_train, x_test, y_test, train_feat, test_feat

def feature_scaling(data):
    a = -1
    b = 1
    _min = []
    _max = []
    _ratio = []
    for col in range(3):
        _min.append(min(data[:,col]))
        _max.append(max(data[:,col]))
        _ratio.append((b-a)/(_max[col]-_min[col]))

        data[:,col] = a + (data[:,col]-_min[col])*_ratio[col]

    data = np.array(data)
    return data, [a, b, _min, _max, _ratio]

def scale_input(data, feature):
    a = feature[0]
    b = feature[1]
    _min = feature[2]
    _max = feature[3]
    _ratio = feature[4]

    for col in range(3):
        data[:,col] = a + (data[:,col]-_min[col])*_ratio[col]

    return data
def normalize_output(scaled_output,feature):
    a = feature[0]
    b = feature[1]
    _min = feature[2]
    _max = feature[3]
    _ratio = feature[4]

    out = []

    for row in range(len(scaled_output)):

        x = ((scaled_output[row,0]-a)/(_ratio[3]))+_min[3]
        y = ((scaled_output[row,1]-a)/(_ratio[4]))+_min[4]
        z = ((scaled_output[row,2]-a)/(_ratio[5]))+_min[5]
        out.append([x,y,z])

    out = np.array(out)

    return out

# def kfold(data, k=10):
#     split = int(len(data)/k)
#     np.random.shuffle(data)
#     test = data[:split,:]

def split_data(test_ratio = 0.2):

    data = np.loadtxt('data/FM_dataset.dat')
    length = int(len(data))
    test_split = int(length*test_ratio)

    np.random.shuffle(data)

    test = data[:test_split,:]
    train = data[test_split:,:]

    f = open('data/test_origin.dat', 'w')
    for row in range(len(test)):
        for col in range(len(test[row])):
            f.write(str(test[row][col]) + "\t\t")
        f.write("\n")
    f.close()

    f = open('data/train_origin.dat', 'w')
    for row in range(len(data)):
        for col in range(len(data[row])):
            f.write(str(data[row][col]) + "\t\t")
        f.write("\n")
    f.close()

    return train, test
