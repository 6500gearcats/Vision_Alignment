import tensorflow.compat.v1 as tf
import numpy as np
import constants as con
import time

tf.disable_v2_behavior() 

x_dataX = [[]]
x_dataY = [[]]
# Muist initialize output_disposables so that the robot can make the initial adjustment
y_output_disposeX = [[0, 0, 0, 0]]
y_output_disposeY = [[0, 0, 0, 0]]
y_dataX = [[1.2, 2.1, 3.1, 4.]]
y_dataY = [[1., 2., 3., 4.]]

# Dataset
# This takes the image and writes it to the text file
def pull_input_data():
    p = 0

def set_ouput_arr():
    pull_input_data()
    input = open(con.Constants.input_data_path, 'r')
    x_y = []
    with open(con.Constants.input_data_path, "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
        x_y = f_list
    
    for i in range(con.Constants.set_size):
        if i % 2 != 0:
            y_output_disposeY[0].append(x_y[i])
        else:
            y_output_disposeX[0].append(x_y[i])

def set_input_data():
    input = open(con.Constants.input_data_path, 'r')
    x_y = []
    with open(con.Constants.input_data_path, "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
        x_y = f_list
    for i in range(con.Constants.set_size):
        if i % 2 != 0:
            x_dataX[0].append(x_y[i])
        else:
            x_dataY[0].append(x_y[i])

# # gets accuracy of new image
def get_acc(x_or_y, outputs):
    net_acc = 0
    iterater = 0
    if x_or_y == 'y':
        for i in range(int(con.Constants.set_size / 2)):
            net_acc = (pow(y_dataY[0][i] - y_output_disposeY[0][i], 2))
            if net_acc < .03 and net_acc > -.03:
                iterater = iterater + 1
    if x_or_y == 'x':
        for i in range(int(con.Constants.set_size / 2)):
            net_acc = (pow(y_dataX[0][i] - y_output_disposeY[0][i], 2))
            if net_acc < .03 and net_acc > -.03:
                iterater = iterater + 1
    
    if iterater == con.Constants.set_size / 2:
        return True
    else:
        return False

def move(out, x_or_y):
    idx = 0
    for i in range(con.Constants.n_output - 1):
        if out[i] < out[i+1]:
            idx = i + 1
    # use idx to be interpreted to move the robot in a particular direction
    
    if x_or_y == 'y':
        time.sleep(0.05)
        p=0
    else:
        time.sleep(0.05)
        p=0

def writeToFile(variable_matX, bias_matX, variable_matY, bias_matY, writeTo):
    for n in range(len(variable_matX)):
        for i in range(con.Constants.l_arr[n]):
            for j in range(con.Constants.l_arr[n+1]):
                writeTo.write(str(variable_matX[n][i][j]))
                writeTo.write(" ")
            writeTo.write("\n")
        for i in range(con.Constants.l_arr[n+1]):
            writeTo.write(str(bias_matX[n][i]))
            writeTo.write(" ")
        writeTo.write("\n")
    writeTo.write("\n")
    for n in range(len(variable_matY)):
        for i in range(con.Constants.l_arr[n]):
            for j in range(con.Constants.l_arr[n+1]):
                writeTo.write(str(variable_matY[n][i][j]))
                writeTo.write(" ")
            writeTo.write("\n")
        for i in range(con.Constants.l_arr[n+1]):
            writeTo.write(str(bias_matY[n][i]))
            writeTo.write(" ")
        writeTo.write("\n")
    writeTo.close()


# Placeholders
Y_x = tf.placeholder(tf.float32, shape=[1, con.Constants.n_input])
X_x = tf.placeholder(tf.float32, shape=[1, con.Constants.n_output])

Y_y = tf.placeholder(tf.float32, shape=[1, con.Constants.n_input])
X_y = tf.placeholder(tf.float32, shape=[1, con.Constants.n_output])

theta1_y = tf.Variable(tf.random_uniform([con.Constants.n_input, con.Constants.n_hidden1], -1, 1))
theta2_y = tf.Variable(tf.random_uniform([con.Constants.n_hidden1, con.Constants.n_hidden2], -1, 1))
theta3_y = tf.Variable(tf.random_uniform([con.Constants.n_hidden2, con.Constants.n_hidden3], -1, 1))
theta4_y = tf.Variable(tf.random_uniform([con.Constants.n_hidden3, con.Constants.n_output], -1, 1))

bias1_y = tf.Variable(tf.zeros([con.Constants.n_hidden1]))
bias2_y = tf.Variable(tf.zeros([con.Constants.n_hidden2]))
bias3_y = tf.Variable(tf.zeros([con.Constants.n_hidden3]))
bias4_y = tf.Variable(tf.zeros([con.Constants.n_output]))

L2_y = tf.sigmoid(tf.matmul(X_y, theta1_y) + bias1_y)
L3_y = tf.sigmoid(tf.matmul(L2_y, theta2_y) + bias2_y)
L4_y = tf.sigmoid(tf.matmul(L3_y, theta3_y) + bias3_y)
hy_y = tf.sigmoid(tf.matmul(L4_y, theta4_y) + bias4_y)

theta1_x = tf.Variable(tf.random_uniform([con.Constants.n_input, con.Constants.n_hidden1], -1, 1))
theta2_x = tf.Variable(tf.random_uniform([con.Constants.n_hidden1, con.Constants.n_hidden2], -1, 1))
theta3_x = tf.Variable(tf.random_uniform([con.Constants.n_hidden2, con.Constants.n_hidden3], -1, 1))
theta4_x = tf.Variable(tf.random_uniform([con.Constants.n_hidden3, con.Constants.n_output], -1, 1))

bias1_x = tf.Variable(tf.zeros([con.Constants.n_hidden1]))
bias2_x = tf.Variable(tf.zeros([con.Constants.n_hidden2]))
bias3_x = tf.Variable(tf.zeros([con.Constants.n_hidden3]))
bias4_x = tf.Variable(tf.zeros([con.Constants.n_output]))

L2_x = tf.sigmoid(tf.matmul(X_x, theta1_x) + bias1_x)
L3_x = tf.sigmoid(tf.matmul(L2_x, theta2_x) + bias2_x)
L4_x = tf.sigmoid(tf.matmul(L3_x, theta3_x) + bias3_x)
hy_x = tf.sigmoid(tf.matmul(L4_x, theta4_x) + bias4_x)

cost_x = tf.reduce_mean(tf.square(Y_x - hy_x))
train_step_x = tf.train.GradientDescentOptimizer(con.Constants.learning_rate).minimize(cost_x)

cost_y = tf.reduce_mean(tf.square(Y_y - hy_y))
train_step_y = tf.train.GradientDescentOptimizer(con.Constants.learning_rate).minimize(cost_y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    pull_input_data()
    set_input_data()
    out_x = sess.run(hy_x, feed_dict={X_x: x_dataX, Y_x: y_output_disposeX})
    move(out_x[0], 'x')
    out_y = sess.run(hy_y, feed_dict={X_y: x_dataY, Y_y: y_output_disposeY})
    move(out_y[0], 'y')

    for i in range(100):
        y_output_disposeX[0].clear()
        y_output_disposeY[0].clear()
        pull_input_data()
        set_ouput_arr()
        if get_acc('x', y_output_disposeX[0]):
            break
        # Update
        _, c_x = sess.run([train_step_x, cost_x], feed_dict={X_x: x_dataX, Y_x: y_output_disposeX})
        _, c_y = sess.run([train_step_y, cost_y], feed_dict={X_y: x_dataY, Y_y: y_output_disposeY})

        print("cost x: ", c_x)
        print("cost y: ", c_y)

        x_dataX[0].clear()
        x_dataY[0].clear()
        set_input_data()
        out_x = sess.run(hy_x, feed_dict={X_x: x_dataX, Y_x: y_output_disposeX})
        move(out_x[0], 'x')
        out_y = sess.run(hy_y, feed_dict={X_y: x_dataY, Y_y: y_output_disposeY})
        move(out_y[0], 'y')

    # Write variables to a file
    writeTo = open(con.Constants.variabale_write_file, 'w')
    mat_arr_x = [sess.run(theta1_x), sess.run(theta2_x), sess.run(theta3_x), sess.run(theta4_x)]
    mat_arr_y = [sess.run(theta1_y), sess.run(theta2_y), sess.run(theta3_y), sess.run(theta4_y)]
    bias_arr_x = [sess.run(bias1_x), sess.run(bias2_x), sess.run(bias3_x), sess.run(bias4_x)]
    bias_arr_y = [sess.run(bias1_y), sess.run(bias2_y), sess.run(bias3_y), sess.run(bias4_y)]
        # Run num layers times (4)
    writeToFile(mat_arr_x, bias_arr_x, mat_arr_y, bias_arr_y, writeTo)
