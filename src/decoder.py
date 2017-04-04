#!/usr/bin/python
#-*-coding:utf-8-*-

import tensorflow as tf
from datasets import datasets

data_sets = datasets()
data_sets.read_data_sets(".", True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 5000])
W = tf.Variable(tf.zeros([5000, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-10))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#training
#tf.global_variables_initializer().run()

saver = tf.train.Saver()
#for i in range(1000):
#	batch_xs, batch_ys = data_sets.train.next_batch(100)
#	train_step.run({x: batch_xs, y_: batch_ys})
	#print cross_entropy
saver.restore(sess, "./model/model.md")
print W.eval()
print b.eval()


# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(acc.eval({x: data_sets.test.text, y_: data_sets.test.label}))


