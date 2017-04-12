#!/usr/bin/python
#-*-coding:utf-8-*-

LAYER_NODE1 = 400 # layer1 node num
INPUT_NODE = 2583 
OUTPUT_NODE = 10

import tensorflow as tf
import nn_interface
from datasets import datasets

def interface(inputs, w1, b1, w2,b2, w3, b3):
	"""
		compute forword progration result
	"""
	lay1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
	lay2 = tf.nn.relu(tf.matmul(lay1, w2) + b2)
	return tf.nn.softmax(tf.matmul(lay2, w3) + b3) # need softmax??

data_sets = datasets()
data_sets.read_test_data(".", True)
#data_sets.read_train_data(".", True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

y = nn_interface.interface(x, None)

saver = tf.train.Saver()
saver.restore(sess, "./model5/model.md")

# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(acc.eval({x: data_sets.test.text, y_: data_sets.test.label}))
#print(acc.eval({x: data_sets.train.text, y_: data_sets.train.label}))


