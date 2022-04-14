#!/bin/bash

pathprefix=$1
i=$2
upper=$(($3+1))
targetfile=$4

while [ $i -ne $((upper)) ]
do
    size=${#i}
    if [ $size -ne 2 ]
    then
        pathstr=$pathprefix/0$i/wsj_0$i*
    else
        pathstr=$pathprefix/$i/wsj_$i*
    fi
    if [ $i == $2 ]
    then
        cat $pathstr > $targetfile
    else
        cat $pathstr >> $targetfile
    fi
    i=$(($i+1))
done


