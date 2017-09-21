#!/bin/bash

function max()
{
    local numA=$1
    local numB=$2
    echo `awk -v numa=$numA -v numb=$numB 'BEGIN{print(numa-numb>0.0001)? numa : numb }'`
}

function get_complete_iter()
{
    local data_dir=${1:-data}
    local _Max=0
    for file in `\ls $data_dir | grep iter | grep pass`
    do
	num=`echo ${file%_*} | awk -F "iter" '{print $2}'`
	_Max=`max $_Max $num`
	#echo $file $num  $_Max  
    done
    echo "$_Max"
}

