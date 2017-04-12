#!/usr/bin/python
#-*-coding:utf-8-*-

LAYER_NODE1 = 500 # layer1 node num
INPUT_NODE = 2583
OUTPUT_NODE = 10
REG_RATE = 0.01

import tensorflow as tf
from datasets import datasets

def interface(inputs, w1, b1, w2,b2):
	"""
		compute forword progration result
	"""
	lay1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
	return tf.nn.softmax(tf.matmul(lay1, w2) + b2) # need softmax??

data_sets = datasets()
data_sets.read_train_data(".", True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE1], stddev=0.1))
b1 = tf.Variable(tf.constant(0.0, shape=[LAYER_NODE1]))

w2 = tf.Variable(tf.truncated_normal([LAYER_NODE1, OUTPUT_NODE], stddev=0.1))
b2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]))

y = interface(x, w1, b1, w2, b2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-10))
regularizer = tf.contrib.layers.l2_regularizer(REG_RATE)
regularization = regularizer(w1) + regularizer(w2)
loss = cross_entropy + regularization


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#training
tf.global_variables_initializer().run()
saver = tf.train.Saver()

cv_feed = {x: data_sets.cv.text, y_: data_sets.cv.label}
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(5000):
	if i % 200 == 0:
		cv_acc = sess.run(acc, feed_dict=cv_feed)
		print "train steps: %d, cv accuracy is %g " % (i, cv_acc)
	batch_xs, batch_ys = data_sets.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

path = saver.save(sess, "./model4/model.md")


