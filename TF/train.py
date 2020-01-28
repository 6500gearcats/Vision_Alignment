import tensorflow.compat.v1 as tf
import numpy as np
import constants as con

tf.disable_v2_behavior() 

# Dataset
x_data = np.array([
[0.,0.], [0.,1.], [1.,0.], [1.,1.]
])
y_data = np.array([
[0., 1.], [1., 0.], [1., 0.], [0., 1.]
])

display_step = 1

# Placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights
W1 = tf.Variable(tf.random_uniform([con.Constants.n_input, con.Constants.n_hidden1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([con.Constants.n_hidden1, con.Constants.n_hidden2], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([con.Constants.n_hidden2, con.Constants.n_hidden3], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([con.Constants.n_hidden3, con.Constants.n_output], -1.0, 1.0))




# Bias
b1 = tf.Variable(tf.zeros([con.Constants.n_hidden1]))
b2 = tf.Variable(tf.zeros([con.Constants.n_hidden2]))
b3 = tf.Variable(tf.zeros([con.Constants.n_hidden3]))
b4 = tf.Variable(tf.zeros([con.Constants.n_output]))

L2 = tf.tanh(tf.matmul(X, W1) + b1)
L3 = tf.cosh(tf.matmul(L2, W2) + b2)
L4 = tf.asinh(tf.matmul(L3, W3) + b3)
hy = tf.sigmoid(tf.matmul(L4, W4) + b4)
cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y) * tf.log(1-hy))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hy, Y))
# cost = -1 * tf.reduce_sum(Y * tf.log(hy))

optimizer = tf.train.GradientDescentOptimizer(con.Constants.learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(con.Constants.epochs):
        _, c = sess.run([optimizer, cost], feed_dict = {X: x_data, Y: y_data})

        if step % display_step == 0:
            print("Cost: ", c)

    answer = tf.equal(tf.floor(hy + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))
    out = sess.run([hy], feed_dict = {X: x_data, Y: y_data})

    for i in range(4):
        print("Predicted: [", round(out[0][i][0], 4), ", ", round(out[0][i][1], 4), "]    Real: ", y_data[i])

    sess.close()   

def get_weight_mat(matrix):
    init_run = tf.Session()
    init_run.run(tf.global_variables_initializer())
    return init_run.run(matrix)

def write_to_txt():
    f = open("out.txt", "a")
    out = get_weight_mat(W1)
    for i in range(2):
        f.write("\n")
        for j in range(5):
            f.write(str(out[i][j]))
            f.write(" ")

    out = get_weight_mat(W2)
    for i in range(5):
        f.write("\n")
        for j in range(8):
            f.write(str(out[i][j]))
            f.write(" ")

    out = get_weight_mat(W3)
    for i in range(8):
        f.write("\n")
        for j in range(5):
            f.write(str(out[i][j]))
            f.write(" ")

    out = get_weight_mat(W4)
    for i in range(5):
        f.write("\n")
        for j in range(2):
            f.write(str(out[i][j]))
            f.write(" ")

    f.close()

write_to_txt()



