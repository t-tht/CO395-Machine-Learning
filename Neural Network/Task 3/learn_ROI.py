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
from keras.models import model_from_json

def main():
    #preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range=(0,2))
    dataset = np.loadtxt("ROI_dataset.dat") #15626 samples
    dataset = scaler.fit_transform(dataset)
    random.shuffle(dataset)
    
    
    #seperation of training and test data
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
    epoch_size= 200
    #the sizes of each batch when running through the training data
    batch_size= 25
    #the split between the training and validation/test data
    validation_ratio=0.1
    test_size= batch_size
    #building the sequential model
    model=Sequential()
    model.add(Dense(35, input_dim=3, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.05))
    #Output layer
    model.add(Dense(4, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_size,
                    verbose=1,
                    validation_split=validation_ratio)
                    
    
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save the model
    nn_json = model.to_json()
    with open ("nn_model.json", "w") as json:
        json.write(nn_json)
    model.save_weights("nn_weights.h5")
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_ROI(network, prep)
    evaluate_architecture(history)

def evaluate_architecture(history):
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

def predict_hidden(dataset):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,2))
    scaled_dataset = scaler.fit_transform(dataset)
    
    # Load the model
    with open('nn_model.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights("nn_weights.h5")

    prediction = model.predict(dataset)

    return prediction


if __name__ == "__main__":
    main()
