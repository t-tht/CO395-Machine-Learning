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

def create_model(neurons=1):
    num_classes=4
    #building the sequential model
    model=Sequential()
    model.add(Dense(neurons, input_dim=3, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.05))
    #model.add(Dense(num_classes *8, activation='relu'))
    #model.add(Dropout(0.2))
    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    #one hot coding hence loss calculated using categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
#    score = model.evaluate(x_test, y_test, verbose=1)
#    prediction=model.predict(x_test)
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])
    # Save the model
#    nn_json = model.to_json()
#    with open ("nn_model1.json", "w") as json:
#    json.write(nn_json)
#    model.save_weights("nn_weights1.h5")

    return model

def main():
    f= open("outputneurons.txt","w+")
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
    
    model =KerasClassifier(build_fn=create_model, epochs=200, batch_size=25, validation_split=0.1, verbose=0)
    
    #batch_size = [10]
    neurons = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    #epoch_size = [100,150,200]
    #optimizer = ['RMSprop', 'SGD']
    param_grid = dict(neurons=neurons)
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
    #layerarray = [["relu",0.2],["relu",0.2]]

    #param_grid = [{"layer_type" : ["relu","sigmoid","softmax","elu","selu","softplus","softsign","tanh","hard_sigmoid","exponential","linear"]}]
    f.close()

def evaluate_architecture(history,prediction):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.hist(prediction_rank)
    plt.show()


def predict_hidden(dataset):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,2))
    scaled_dataset = scaler.fit_transform(dataset)
    
    # Load the model
    with open('nn_model.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_wights("nn_weights.h5")

    prediction = model.predict(dataset)

    return prediction


if __name__ == "__main__":
    main()
