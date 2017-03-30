#!/usr/bin/python
#-*-coding:utf-8-*-
import jieba
import os
from contextlib import nested

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class dataset():
	"""
		data preparation class,
		1. text split using jieba
		2. delete same and stop words.
		2. generate dict.
		3. generate data(test and train) vector.
	"""
	def __init__(self):
		pass

	def get_unique_id(self, data_dir):
		"""
			get flie unique id famate as {class_id}_type_{text_id}.txt.
			data_dir is the full path of file
		  	e.g ./training_set/4_tec/4_tec_text/text_2001.txt
			where "training" is type, "4" is file class, and "2001" is text id.
			modify this function to adapt your data dir fomate
		"""
		dir_list = data_dir.split("/")
		class_id = dir_list[2].split("_")[0]
		text_id = dir_list[4].split(".")[0]
		type_id = dir_list[1].split("_")[0]
		return class_id + "_" + type_id + "_" + text_id


	def splitwords(self, data_dir, data_type):
		""" 
			split word for all files under data_dir 
			save data as <class_{data_type}_id> <words> in ./{data_type}_file2words.txt,
			where data_type is train, test or cv.
		"""
		os.remove(data_type+".txt")
		list_dirs = os.walk(data_dir)
		for root, _, files in list_dirs:
			# get all files under data_dir
			for fp in files:
				file_path = os.path.join(root, fp)
				file_id = self.get_unique_id(file_path)
				#split words for f, save in file ./data_type.txt
				with nested(open(file_path), open(data_type+".txt", "a+")) as (f1, f2):
					data = f1.read()
					#print data
					seg_list = jieba.cut(data, cut_all=False)
					f2.write(file_id + " " + " ".join(seg_list).replace("\n", " ")+"\n")
					break
		print "split " + data_type + " end.\n"

if __name__ == '__main__':
	data_pre = DataPre()
	data_pre.splitwords("../training_set", "train")
