import pickle
import tensorflow as tf
import numpy as np
from mnist import MNIST
one_hot_saved = 1
# ~~~~~-~~~~~ DEFINE LAYERS ~~~~~-~~~~~
# Simple Convolutional layer
def conv_layer(input_img, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]))
        conv = tf.nn.conv2d(input_img, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("actuations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# FC layer
def fc_layer(input_frame, channels_in, channels_out, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.2))
        b = tf.Variable(tf.constant(0.05,shape=[channels_out]))
        act = tf.nn.relu(tf.matmul(input_frame, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("actuations", act)
        return act

tf.reset_default_graph()
with tf.Session() as sess:
# ~~~~~-~~~~~ DEFINE INPUTS ~~~~~-~~~~~
# Setup placerholders and reshape the data
    #tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    print("Defined placeholders")

# ~~~~~-~~~~~ FEED-FORWARD STEP ~~~~~-~~~~~
# Network
    conv1 = conv_layer(x_image, 1, 32, name="conv1")
    conv2 = conv_layer(conv1, 32, 64, name="conv2")
    flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name="fcl")
    relu = tf.nn.relu(fc1)
    tf.summary.histogram("fc1/relu", relu)
    logits = fc_layer(relu, 1024, 10, name="fc2")
    print("Defined network")

# ~~~~~-~~~~~ LOSS & TRAINING ~~~~~-~~~~~
# Compute cross entropy as our loss function
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Use an AdamOptimizer to train the network
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        tf.summary.scalar("xent", cross_entropy)

# compute accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    print("Defined post-network stuff")

# ~~~~~-~~~~~ Train Model ~~~~~-~~~~~

# Initialize all the variables
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Define tensorboard writer class
    writer = tf.summary.FileWriter("./log/1")
    writer.add_graph(sess.graph)
    mndata = MNIST('/home/umair/python-mnist/data/')
    images, labels_raw = mndata.load_training()
    labels = []
    print("Defined data")
    if(one_hot_saved):
	with open("one_hot.pckle",'rb') as fili:
	    labels = pickle.load(fili)
    else:
	for i in range(len(labels_raw)):
	    tmp = [0]*10
	    tmp[labels_raw[i]] = 1
	    labels = labels + [tmp]
	with open("one_hot.pckle",'wb') as filp:
	    pickle.dump(labels,filp)
    # Train for 2000 steps
    print("Defined one-hot data")
    batch_size = 50
    for i in range(50):
        indices = np.random.randint(0,len(images),batch_size)
        image_b = [images[ind] for ind in indices]
        labels_b = [labels[ind] for ind in indices]
        #print("loaded iteration "+str(i)+" data")
        # report accuracy
        if i % 5 == 0:
            [train_accuracy,s] = sess.run([accuracy, merged_summary_op], feed_dict={x: image_b, y: labels_b})
	    writer.add_summary(s,i)
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # run the training step
        sess.run(train_step, feed_dict={x: image_b, y: labels_b})
#    writer.close()
        # USE tensorboard --logdir ./tmp/1

        # 
        # tf.summary.image('input',x_image,3)
