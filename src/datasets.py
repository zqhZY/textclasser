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

	def __init__(self):
		"""
			init
		"""
		pass

	def read_datasets(self):
		"""
			called to read all data, {train.txt test.txt cv.txt}

		"""
		pass


class datasets(object):
	"""
		read datasets from {train.txt test.txt cv.txt}
	"""

	def __init__(self):
		pass

	def read_data_sets(self, data_dir, one_hot=False, dtype=tf.uint8):
		"""
			read datasets from {train.txt test.txt cv.txt}
			save as numpy array such as train_texts and train_lable
		"""
		pass
		
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
				if text_num == 2:
					break
			
			text = numpy.array(tmp_text)
			text_dim = len(tmp_text)/text_num
			text = text.reshape((text_num, text_dim))
			
			label_num = 0
			for line in f2:
				label_num += 1
				label = map(int, line.split())
				tmp_label.extend(label)
				if label_num == 2:
					break

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
		labels_one_hot = numpy.zeros((label_num, class_num))
		labels_one_hot.flat[offset+labels.ravel()] = 1
		return labels_one_hot

		
if __name__ == '__main__':
	d = datasets()
	d.read_from_disk(".", "train", True)
