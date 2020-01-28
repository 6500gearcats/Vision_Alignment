# Has the constants for the neural network
# Epochs - number of time the feedForward is ran
# Batches - the number of times an epoch is ran before an update
# n_input - number of neurons in input layer
# n_hidden1, n_hidden2, ... - number of neurons in hidden layers
# n_output - number of neurons in output layer
# learning_rate - the rate at which weights are updated

class Constants:
    epochs = 10000
    batches = 2
    n_input = 2
    n_hidden1 = 5
    n_hidden2 = 8
    n_hidden3 = 5
    n_output = 2
    learning_rate = 0.05