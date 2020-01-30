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

    num_sets = 1

    # contains x and y data
    set_size = 8

    n_input = 4
    n_hidden1 = 5
    n_hidden2 = 8
    n_hidden3 = 5

    # The outputs are move up, move down, turn left, turn right
    n_output = 4
    learning_rate = .1

    l_arr = [n_input, n_hidden1, n_hidden2, n_hidden3, n_output]
    variabale_write_file = '/Users/idler/Desktop/GitHub/Vision_Alignment/cpp_api/out.txt'
    input_data_path = '/Users/idler/Desktop/GitHub/Vision_Alignment/TF/model_data/input.txt'
    target_data_path = '/Users/idler/Desktop/GitHub/Vision_Alignment/TF/model_data/target_data.txt'
