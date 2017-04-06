#!/bin/bash
#
# change dir/* text format from doc to unix
# some file edit in windows may not display or
# process in unix env. change file format using
# doc2unix commend.
# $1 is the dir that all files under this dir will
# be processed
#

function doc2unix(){
	for file in `ls $1`
	do
		if [ -d $1 ]
		then
			doc2unix $1/$file
		else
			dos2unix $file
		fi
	done
}

doc2unix $1
