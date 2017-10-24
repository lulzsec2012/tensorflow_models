#!/bin/bash
trap "kill 0" INT
export CUDA_VISIBLE_DEVICES=0

############################Configs############################
MODEL_NAME=alexnet_v2
TRAIN_DIR=train_logs_alexnet
DATASET_NAME=flowers102
NUM_CLONES=1
###############################################################

if [ -d /run/shm ]
then
    shm_dir=/run/shm
else
    shm_dir=/tmp
fi
echo "shm_dir="$shm_dir

if [ ! -d $shm_dir/$DATASET_NAME -a -d /home/lzlu/dataset/$DATASET_NAME ]
then
    cp /home/lzlu/dataset/$DATASET_NAME $shm_dir/ -r
fi
DATASET_DIR=$shm_dir/$DATASET_NAME

#DATASET_SPLIT_NAME_FOR_VAL=validation  #for vgg




if [ $DATASET_NAME = "cifar10" -o $DATASET_NAME = "mnist" ]
then
    DATASET_SPLIT_NAME_FOR_VAL=test #for mnist
    LABELS_OFFSET=0  
elif [ $DATASET_NAME = "flowers102" ]
then
    DATASET_SPLIT_NAME_FOR_VAL=validation #for mnist
    LABELS_OFFSET=0  #vgg resnet 1000+1 (1 for background)
else
    :
fi


function next_CHECKPOINT_PATH()
{
    local train_dir=$1    
    if [ ! -f $train_dir/checkpoint ]
	then
	echo "Error:File $train_dir/checkpoint do not exit!"
	exit -1
    fi
    local ckpt_num=`cat $train_dir/checkpoint | grep -v all_model_checkpoint_paths | awk -F "-" '{print $2}' | awk -F "\"" '{print $1}'` 
    checkpoint_path=$train_dir/model.ckpt-$ckpt_num
    echo $checkpoint_path
}



for((s=1;s<=100;s+=1))
do
    let "steps=s*2000"
    echo "XXXsteps="$steps
    if [ -d $TRAIN_DIR/steps$steps ]
    then
	continue
    fi

    #./eval.sh 
    echo "python3 ../train_image_classifier.py --train_dir=$shm_dir/${TRAIN_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --dataset_dir=${DATASET_DIR} --model_name=$MODEL_NAME --pruning_gradient_update_ratio=0 --max_number_of_steps=$steps  --learning_rate=0.001 --learning_rate_decay_type=fixed --save_interval_secs=200 --log_every_n_steps=50 --batch_size=64 --labels_offset=$LABELS_OFFSET"
    python3 ../train_image_classifier.py --train_dir=$shm_dir/${TRAIN_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --dataset_dir=${DATASET_DIR} --model_name=$MODEL_NAME --pruning_gradient_update_ratio=0 --max_number_of_steps=$steps  --learning_rate=0.001 --learning_rate_decay_type=fixed --save_interval_secs=200 --log_every_n_steps=50 --batch_size=64 --labels_offset=$LABELS_OFFSET 
    checkpoint_path=`next_CHECKPOINT_PATH $shm_dir/${TRAIN_DIR}`
    echo "checkpoint_path="$checkpoint_path
    echo "python3 ../eval_image_classifier.py  --checkpoint_path=$checkpoint_path --eval_dir=${TRAIN_DIR}/eval --alsologtostderr --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50 --labels_offset=$LABELS_OFFSET 2>&1 | grep logging     "
    python3 ../eval_image_classifier.py  --checkpoint_path=$checkpoint_path --eval_dir=${TRAIN_DIR}/eval --alsologtostderr --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50 --labels_offset=$LABELS_OFFSET 2>&1 | grep logging     
    mkdir -p $TRAIN_DIR
    cp $shm_dir/$TRAIN_DIR $TRAIN_DIR/steps$steps -r
done

