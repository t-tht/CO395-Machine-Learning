learn_FM.py

contains two functions: predict_hidden() and evaluate_architecture()

predict_hidden(hidden_data):
    - Takes in hidden_data that is in the same for as "FM_dataset.dat"
    - Outputs:
        1. y_pred   : predicted output based on the input
        2. model    : model object that describes the neural net
        3. history  : history object that describes the training period
        4. param    : the best performing set of hyperparameters
        5. train_feat : training features required for scaling

evaluate_architecture(hidden_data, y_pred, model, history, param, train_feat):
    - Inputs:
        1.  hidden_data :   test data
        2.  y_pred      :   predicted output
        2. model    : model object that describes the neural net
        3. history  : history object that describes the training period
        4. param    : the best performing set of hyperparameters
        5. train_feat : training features required for scaling