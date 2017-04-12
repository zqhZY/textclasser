#!/usr/bin/python
#-*-coding:utf-8-*-

LAYER_NODE1 = 400 # layer1 node num
INPUT_NODE = 2583
OUTPUT_NODE = 10
REG_RATE = 0.01

MODEL_PATH="./model"


import tensorflow as tf
import nn_interface
from datasets import datasets

def train(data_sets):
	"""
		train model
	"""
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")


	reg = tf.contrib.layers.l2_regularizer(REG_RATE)
	y = nn_interface.interface(x, reg)

	cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-10))
	loss = cross_entropy + tf.add_n(tf.get_collection("losses"))

	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	saver = tf.train.Saver()

	cv_feed = {x: data_sets.cv.text, y_: data_sets.cv.label}
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for i in range(20000):
			if i % 200 == 0:
				cv_acc = sess.run(acc, feed_dict=cv_feed)
				print "train steps: %d, cv accuracy is %g " % (i, cv_acc)
			batch_xs, batch_ys = data_sets.train.next_batch(50)
			train_step.run({x: batch_xs, y_: batch_ys})
		path = saver.save(sess, "./model5/model.md")

def main(argv=None):
	data_sets = datasets()
	data_sets.read_train_data(".", True)
	train(data_sets)


if __name__ == "__main__":
	tf.app.run()



