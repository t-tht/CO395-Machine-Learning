import numpy as np
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
import random
from sklearn import preprocessing


def create_model():
    #epoch is how many times the all of the training data is run through
    #epoch_size= 100
    #the sizes of each batch when running through the training data
    #batch_size= 64
    #the split between the training and validation/test data for training
    num_classes=4
    #building the sequential model
    model=Sequential()
    model.add(Dense(num_classes *8, input_dim=3, kernel_initializer='random_uniform', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(num_classes *8, activation='relu'))
    #model.add(Dropout(0.2))
    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    #one hot coding hence loss calculated using categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

def main():
    f= open("outputbatch.txt","w+")
    scaler = preprocessing.MinMaxScaler(feature_range=(0,2))
    dataset = np.loadtxt("ROI_dataset.dat") #15626 samples
    dataset = scaler.fit_transform(dataset)
    random.shuffle(dataset)
    #for training, features, first 14000 samples
    x_train=dataset[:-1626,:3]
    #for training, labels, first 14000 samples
    y_train=dataset[:-1626,3:]
    #for testing, features, last 1626 samples
    x_test=dataset[-1626:, :3]
    #for testing, labels, last 1626 samples
    y_test=dataset[-1626:,3:]
    
    model =KerasClassifier(build_fn=create_model, epochs=150, batch_size=15, validation_split=0.1, verbose=0)
    batch_size = [5,10,15,20,25]
    
    epoch_size = [100,150,200, 250]
    param_grid = dict(batch_size=batch_size, epochs=epoch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    f.write("\n"+"Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f, %f, %r" % (mean, stdev, param))
        f.write("\n"+"%f, %f, %r" % (mean, stdev, param))
    f.close()


if __name__ == "__main__":
    main()
