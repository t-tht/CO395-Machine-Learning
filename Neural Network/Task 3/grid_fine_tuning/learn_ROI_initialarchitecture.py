import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import random

#from nn_lib import (
#    MultiLayerNetwork,
#    Trainer,
#    Preprocessor,
#    save_network,
#    load_network,
#)

#from illustrate import illustrate_results_ROI


def main():
    dataset = np.loadtxt("ROI_dataset.dat") #15626 samples
    random.shuffle(dataset)
    #for training, features, first 14000 samples
    x_train=dataset[:-1626,:3]
    #for training, labels, first 14000 samples
    y_train=dataset[:-1626,3:]
    #for testing, features, last 1626 samples
    x_test=dataset[-1626:, :3]
    #for testing, labels, last 1626 samples
    y_test=dataset[-1626:,3:]
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    #epoch is how many times the all of the training data is run through
    epoch_size= 100
    #the sizes of each batch when running through the training data
    batch_size= 64
    #the split between the training and validation/test data for training
    validation_ratio=0.1
    test_size= batch_size
    num_classes=4
    #building the sequential model
    model=Sequential()
    model.add(Dense(num_classes *8, input_dim=3, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes *8, activation='relu' ))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes *8, activation='relu' ))
    model.add(Dropout(0.2))

    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    model.summary()
    #one hot coding hence loss calculated using categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_size,
                    verbose=1,
                    validation_split=validation_ratio,
                    validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    prediction=model.predict(x_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_ROI(network, prep)
    evaluate_architecture(history, prediction)

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


if __name__ == "__main__":
    main()
