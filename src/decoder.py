#!/usr/bin/python
#-*-coding:utf-8-*-

import tensorflow as tf
from datasets import datasets

data_sets = datasets()
data_sets.read_test_data(".", True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 5000])
W = tf.Variable(tf.zeros([5000, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

saver = tf.train.Saver()
saver.restore(sess, "./model2/model.md")
print W.eval()
print b.eval()


# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(acc.eval({x: data_sets.test.text, y_: data_sets.test.label}))


