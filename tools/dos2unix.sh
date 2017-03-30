#!/bin/bash

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
