#!/bin/bash
DIR_IN=$1
DIR_OUT=$DIR_IN-png
mkdir -p $DIR_OUT
for f in $(ls $DIR_IN/*pgm); do
	bname=$(basename $f)
	echo $bname
	magick convert $f $DIR_OUT/${bname/%pgm/png}
done
