#!/usr/bin/python
#-*-coding:utf-8-*-
"""
	PCA for datasets
	
"""
import os
import sys
import commands
import numpy
from contextlib import nested
from datasets import datasets

K_DIM = 1000
ORIGIN_DIM = 5000

def pca(origin_mat):
	"""
		gen matrix using pca
		row of origin_mat is one sample of dataset
		col of origin_mat is one feature
		return matrix  U, s and  V 
	"""
	# mean,normaliza1on
	avg = numpy.mean(origin_mat, axis=0)
	# covariance matrix
	cov = numpy.cov(origin_mat-avg,rowvar=0)
	#Singular Value Decomposition
	U, s, V = numpy.linalg.svd(cov, full_matrices=True)

	k = 1;
	sigma_s = numpy.sum(s)
	# chose smallest k for 99% of variance retained 
	for k in range(1, ORIGIN_DIM+1):
		variance = numpy.sum(s[0:k]) / sigma_s
		print "k = %d, variance is %f" % (k, variance)
		if variance >= 0.99:
			break

	if k == ORIGIN_DIM:
		print "some thing unexpected , k is same as ORIGIN_DIM"
		exit(1)

	return U[:, 0:k], k

if __name__ == '__main__':
	"""
		main, read train.txt, and do pca
		save file to train_pca.txt
	"""
	data_sets = datasets()
	train_text, _ = data_sets.read_from_disk(".", "train", one_hot=False)

	U, k = pca(train_text)
	print "U shpae: ", U.shape
	print "k is : ", k

	text_pca = numpy.dot(train_text, U)
	text_num = text_pca.shape[0]
	print "text_num in pca is ", text_num

	with open("./train_pca.txt", "a+") as f:
		for i in range(0, text_num):
			f.write(" ".join(map(str, text_pca[i,:])) + "\n")
