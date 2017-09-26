#!/bin/bash

function max()
{
    local numA=$1
    local numB=$2
    echo `awk -v numa=$numA -v numb=$numB 'BEGIN{print(numa-numb>0.0001)? numa : numb }'`
}

function get_current_iter_major()
{
    local data_dir=${1:-data}
    local _Max=0
    for file in `\ls $data_dir | grep iter | grep "-"`
    do
	num=`echo $file | awk -F "iter" '{print $2}' | awk -F "-" '{print $1}'`
	_Max=`max $_Max $num`
	#echo $file $num  $_Max  
    done 
    echo "$_Max"
}

function get_current_iter_minor()
{
    local data_dir=${1:-data}
    local iter_major=$2
    local _Max=100
    for file in `\ls $data_dir | grep "iter${iter_major}"`
    do
	num=`echo $file | awk -F "-" '{print $2}'`
	_Max=`max $_Max $num`
	#echo $file $num  $_Max  
    done 
    echo "$_Max"
}

iter_major=`get_current_iter_major data`
echo "iter_major="$iter_major
get_current_iter_minor data $iter_major