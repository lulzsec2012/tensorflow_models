#!/bin/bash

checkpoint_dir=$1
if [ $# -ge 2 ]
then
    train_dir=$2
else
    train_dir=${1}_ckpt
fi

function write_to_file()
{
    filename=$1
    content=$2
    mkdir -p `dirname $filename`
    echo "write_to_file::filename=$1"
    echo "write_to_file::content=$2"
    exec 8<&1
    exec > $filename
    echo $content
    exec 1<&8
    exec 8<&-
}

function read_from_file()
{
    filename=$1
    if ! [ -f $filename ]
    then
        echo "Error:File $filename do not exit!"
        exit -1
    fi
    exec <$filename
    read rates
    echo "$rates"
}




rm $train_dir/Retrain_Prunned_Network -rf
mkdir -p $train_dir/Retrain_Prunned_Network

cp $checkpoint_dir/Retrain_Prunned_Network/ckhp_iter_PC.txt $train_dir/Retrain_Prunned_Network -v
if [ $# -eq 3 ]
then
    write_to_file $train_dir/Retrain_Prunned_Network/ckhp_iter_PC.txt $3 #2>&1 >> /dev/null
fi
iter=`read_from_file "$train_dir/Retrain_Prunned_Network/ckhp_iter_PC.txt"`

cp -rv $checkpoint_dir/Retrain_Prunned_Network/iter${iter}_pass $train_dir/Retrain_Prunned_Network 
