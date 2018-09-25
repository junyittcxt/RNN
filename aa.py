import tensorflow as tf
import numpy as np
 
n_inputs = 1     # number of neurons in input layer
n_neurons = 128  # number of neurons in RNN net
n_seq = 100      # length of training sequences. this is also the back-propagation truncation size
 
# Input layer.shape = [batch_size, sequence_size, input_size]
X = tf.placeholder(tf.float32,shape=[None,n_seq,n_inputs])
# RNN type. This is not a layer, it is a "layer generator".
# You can replace BasicRNNCell for BasicLSTM or other types.
cell = tf.contrib.rnn.BasicRNNCell(n_neurons)
# Generate the actual net.
# outputs.shape = [batch_size, sequence_size, n_neurons]
# states: not used
outputs,states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
# Connect RNN layer to output layer. Bulky because weights are shared.
y = tf.reshape(tf.layers.dense(tf.reshape(outputs,[-1,n_neurons]),n_inputs),[-1,n_seq,n_inputs])
# Prepare for training.
y_target = tf.placeholder(tf.float32,shape=[None,n_seq,n_inputs])
loss = tf.reduce_mean(tf.square(y-y_target),axis=[1,2]) # shape = [None,]
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train_op = optimizer.minimize(loss) 
 
# Train with 2000 episodes, randomly generated.
init = tf.global_variables_initializer()
loss_save = []
with tf.Session() as sess:
    init.run()
    saver = tf.train.Saver()
    for i in range(2000):
        X_batch = np.random.rand(1,n_seq,n_inputs)
        y_batch = np.roll(X_batch,1,axis=1) # right-shift
        loss_val, _ = sess.run([loss,train_op],feed_dict={X:X_batch,y_target:y_batch})
        loss_save.append(loss_val[0])
    saver.save(sess,'tmp.ckpt')
plt.figure(figsize=[12,6])
plt.plot(loss_save)
plt.xlabel('episode')
plt.ylabel('loss')