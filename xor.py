import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Set the seed of the pseudorandom number tenerator
seed = 1
tf.random.set_random_seed(seed)

# There are 2^7=128 input-output pairs
NUM_PAIRS = 4

# The middle layer has 3 units
MIDDLE = 3

# Create placeholders for the inputs and outputs
X = tf.placeholder(dtype=tf.float32, shape=(NUM_PAIRS, 2))
Y = tf.placeholder(dtype=tf.float32, shape=(NUM_PAIRS, 1))

# Set up the inputs
LIST_INPUT = [[0,0],
              [0,1],
              [1,0],
              [1,1]]

# Set up the outputs
LIST_OUTPUT = [[0],
               [1],
               [1],
               [0]]

# Set the learning rate
learning_rate = 0.001

# Set the number of epochs
N = 10000

# Set up the hidden layer
with tf.variable_scope('hidden'):
  
    # Randomize the weights and biases
    h_w = tf.Variable(2*tf.random.uniform([2, MIDDLE])-1, name='weights')
    h_b = tf.Variable(2*tf.random.uniform([NUM_PAIRS, MIDDLE])-1, name='biases')
    
    # Calculate the values of the units
    h = tf.nn.relu(tf.matmul(X, h_w) + h_b)

# Set up the output layer
with tf.variable_scope('output'):
      
    # Randomize the weights and biases
    o_w = tf.Variable(2*tf.random.uniform([MIDDLE, 1])-1, name='weights')
    o_b = tf.Variable(2*tf.random.uniform([NUM_PAIRS, 1])-1, name='biases')
    
    # Calculate the values of the units
    Y_estimation = tf.nn.sigmoid(tf.matmul(h, o_w) + o_b)

# Set up the loss function
with tf.variable_scope('cost'):
  
    # Calculate the average cost
    cost = tf.reduce_mean(tf.squared_difference(Y_estimation, Y))

# Set up the training variable
with tf.variable_scope('train'):
  
    # Train using the Adam optimizer
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Start the TensorFlow session
with tf.Session() as session:
  
    # Initialize the session
    session.run(tf.global_variables_initializer())
    
    for epoch in range(N):
      
        # Train the model
        session.run(train, feed_dict={X: LIST_INPUT, Y:LIST_OUTPUT})

        if epoch % (N/10) == 0:
            # Print the epoch and the average loss
            loss = session.run(cost, feed_dict={X: LIST_INPUT, Y:LIST_OUTPUT})
            print("epoch = {0}, average loss = {1}".format(epoch, loss))

    # Print the outputs
    Y_test = session.run(Y_estimation, feed_dict={X:LIST_INPUT})
    print(np.round(Y_test, decimals=3))
