#!/usr/bin/python
#-*-coding:utf-8-*-

import tensorflow.python.platform

import os
import sys
import numpy
import tensorflow as tf
from contextlib import nested

reload(sys)
sys.setdefaultencoding('utf-8')

class dataset(object):
	"""
		data set info readed in disk file
	"""

	def __init__(self, text, label, dtype=tf.float32):
		"""
			init
		"""
		self._num_examples = text.shape[0]
		self._text = text
		self._label = label
		self._index_in_epoch = 0
		self._epochs_completed = 0

	@property
	def text(self):
		return self._text

	@property
	def label(self):
		return self._label
	
	def next_batch(self, batch_size):
		"""
			get next batch
		"""

		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			
			#complete epochs
			self._epochs_completed += 1
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._text = self._text[perm]
			self._label = self._label[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		
		end = self._index_in_epoch
		return self._text[start:end], self._label[start:end]
	

class datasets(object):
	"""
		read datasets from {train.txt test.txt cv.txt}
	"""

	def __init__(self):
		"""

		"""
		pass

	def read_data_sets(self, data_dir, one_hot=False, dtype=tf.float32):
		"""
			read datasets from {train.txt test.txt cv.txt}
			save as numpy array such as train_texts and train_lable
		"""

		# read training set, text to train.text, label to trian.label
		train_text, train_label = self.read_from_disk(data_dir, "train", one_hot)
		test_text, test_label = self.read_from_disk(data_dir, "test", one_hot)
		
		validation_size = train_text.shape[0] / 8
		print "validation size is %d " % validation_size
		cv_text = train_text[:validation_size]
		cv_label = train_label[:validation_size]
		print cv_text.shape
		print cv_label.shape
		
		train_text = train_text[validation_size:]
		train_label = train_label[validation_size:]

		self.train = dataset(train_text, train_label)
		self.test = dataset(test_text, test_label)
		self.cv = dataset(cv_text, cv_label)

		return
		
	def read_from_disk(self, data_dir, data_type, one_hot=False):
		"""
			read certain data from disk, data are been generated using
			data_prepare.py
		"""
		
		data_path = data_dir + "/" + data_type + ".txt"
		label_path = data_dir + "/" + data_type + "_labels.txt"

		
		with nested(open(data_path), open(label_path)) as (f1, f2):
			tmp_text = []
			tmp_label = []
			text_num = 0
			for line in f1:
				text_num += 1
				# word list form str to int
				words = map(int, line.split())
				tmp_text.extend(words)
				#if text_num == 10000:
				#	break
			
			text = numpy.array(tmp_text)
			text_dim = len(tmp_text)/text_num
			text = text.reshape((text_num, text_dim))
			
			label_num = 0
			for line in f2:
				label_num += 1
				label = map(int, line.split())
				tmp_label.extend(label)
				#if label_num == 10000:
				#	break

			label = numpy.array(tmp_label)
			label[label==10] = 0
			if one_hot:
				label = self.to_one_hot(label)
			#print label.shape
			#label_dim = len(tmp_label) / label_num
			#label = label.reshape((label_num, label_dim))
			assert label_num == text_num, "label num and text num must be equal"
			return text, label
	
	def to_one_hot(self, labels, class_num=10):
		"""
			dense labels to one_hot vectors
		"""
		label_num = labels.shape[0]
		offset = numpy.arange(label_num) * class_num
		labels_one_hot = numpy.zeros((label_num, class_num), dtype=numpy.uint8)
		labels_one_hot.flat[offset+labels.ravel()] = 1
		#print labels_one_hot
		return labels_one_hot

		
if __name__ == '__main__':
	d = datasets()
	d.read_from_disk(".", "train", True)
	#d.read_data_sets(".", True)
	#print d.train.next_batch(1)[1]
