#!/bin/bash
#####################################################
#                 Global Config
#####################################################
DATASET_DIR=/tmp/mnist #/mllib/ImageNet/ILSVRC2012_tensorflow
DATASET_NAME=mnist     #imagenet
TRAIN_DIR_PREFIX=./train_dir_multiLayer
EVAL_INTERVAL=20
SAVE_SUMMARIES_SECS=250
DEFAULT_MAX_NUMBER_OF_STEPS=200
#DATASET_SPLIT_NAME_FOR_VAL=validation  #for vgg
DATASET_SPLIT_NAME_FOR_VAL=test #for mnist
MODEL_NAME=lenet #vgg_16
LABELS_OFFSET=0  #vgg resnet 1000+1 (1 for background)
NUM_CLONES=1

#####################################################
#           Pruning and Retrain Config
#####################################################
checkpoint_path=./mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
TRAIN_DIR_PREFIX=./train_dir_multiLayers_OK_231_xxx
train_Dir=${TRAIN_DIR_PREFIX}_${MODEL_NAME}/Retrain_Prunned_Network
all_trainable_scopes="LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1"

#####################################################
#      Global Flags (may change dure the program)
#####################################################
g_starting_pruning_rate=1.00
g_starting_pruning_step=0.10


#####################################################
#              Algorithm Configs
#####################################################

#*Algorithm:REPRUNING_FROM_SPECIFIC_LAYER*#
En_REPRUNING_FROM_SPECIFIC_LAYER="Disable" #["Enable":"Disable"]
REPRUNING_FROM_LAYER_TH=0

#*Algorithm:AUTO_DRAW_BACK_WHILE_PRUNING*#


#*Algorithm:AUTO_RATE_PRUNING_WITHOUT_TRAINING*#
En_AUTO_RATE_PRUNING_WITHOUT_RETRAIN="Enable" #["Enable":"Disable"]


En_A_Retrain_for_ImageNet="Disable" #["Enable":"Disable"]

#*Algorithm:AUTO_RATE_PRUNING_EARLY_SKIP*#
En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable" #["Enable":"Disable"]





prune_net_args="--noclone_on_cpu --optimizer=sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	--save_summaries_secs=$SAVE_SUMMARIES_SECS --pruning_gradient_update_ratio=0  --pruning_strategy=ABS \
        --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.001 --weight_decay=0.0005 --batch_size=64 \
         --num_clones=$NUM_CLONES" #all_trainable_scopes

evalt_net_args="--alsologtostderr  --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50" #all_trainable_scopes

#Real eval command: python3 train_image_classifier.py --noclone_on_cpu --optimizer=sgd --labels_offset=0 --dataset_dir=/tmp/mnist             --dataset_name=mnist --dataset_split_name=train --model_name=lenet 	    --save_summaries_secs=250 --checkpoint_path=./train_dir_multiLayers_OK_231_xxx_userless_lenet/Retrain_Prunned_Network/iter5_pass/model.ckpt-20 --train_dir=./train_dir_multiLayers_OK_231_xxx_userless_lenet/Retrain_Prunned_Network/iter5 --trainable_scopes=LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1 --pruning_scopes=LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1 --pruning_rates=1.00,1.00,.20,.20 --max_number_of_steps=40 --pruning_strategy=ABS --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.001 --weight_decay=0.0005 --batch_size=64 	    --max_number_of_steps=30 --pruning_gradient_update_ratio=0 --num_clones=1

train_dir=./tmp/prune
checkpoint_path=./mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
check_dir=../mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
max_number_of_steps=100
all_trainable_scopes="LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1"
trainable_scopes=$all_trainable_scopes
pruning_scopes="LeNet/conv2,LeNet/conv1"
pruning_rates="0.8,0.8"

function train_program()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
    local prune_net_args=$7
    local cmd_str="--train_dir=$train_dir --checkpoint_path=$check_dir --max_number_of_steps=$max_number_of_steps\
             --pruning_scopes=$pruning_scopes --pruning_rates=$pruning_rates --trainable_scopes=$trainable_scopes"
    echo $cmd_str $prune_net_args
    
    python3 ../train_image_classifier.py  $cmd_str $prune_net_args
}

#train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_rates train_scopes 
function next_CHECKPOINT_PATH()
{
    local train_dir=$1    
    if [ ! -f $train_dir/checkpoint ]
	then
	echo "Error:File $train_dir/checkpoint do not exit!"
	exit -1
    fi
    _ckpt_=`cat $train_dir/checkpoint | grep -v all_model_checkpoint_paths | awk '{print $2}'`
    ckpt_=${_ckpt_#\"}
    ckpt=${ckpt_%\"}
    if [ ${train_dir:0:1} = "/" ]
    then
	checkpoint_path=$ckpt
    else
	checkpoint_path=$train_dir/$ckpt
    fi
    echo $checkpoint_path
}

function get_Recall_5()
{
    local result_str="$*" #`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Recall_5" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Recall_5() failed!"
	exit -1
    fi
    echo $result_mul_10000
}

function get_Accuracy()
{
    local result_str="$*" #`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Accuracy" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Accuracy() failed!"
	exit -1
    fi
    echo $result_mul_10000
}

#train_dir evalt_loss_anc evalt_loss_thr evalt_loss_drp 
#g_evalt_loss_pass
#train_dir=/tmp/prune
evalt_loss_anc=9900
evalt_loss_drp=50

function evalt_program()
{
    local train_dir=$1
    local tmp_checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
    local edata_dir=${train_dir}/eval_event
    cmd_str="--checkpoint_path=${tmp_checkpoint_path}  --eval_dir=$edata_dir"   
    echo $cmd_str $evalt_net_args
    local result_str=`python3 ../eval_image_classifier.py  $cmd_str $evalt_net_args   2>&1 | grep logging`
    g_Accuracy=`get_Accuracy $result_str`
    g_Recall_5=`get_Recall_5 $result_str`
    echo "g_Accuracy="$g_Accuracy
    echo "g_Recall_5="$g_Recall_5

    g_evalt_loss_pass="False"
    if [ -z "$g_Accuracy" ]
    then
	echo "Error: envoke evalt_program failed!"
	echo "result_str="$result_str
	return 1
    else
        ##compute g_evalt_loss_pass
	evalt_loss_cur=$g_Accuracy
	let "evalt_loss_thr=evalt_loss_anc-evalt_loss_drp"
	if [ $evalt_loss_cur -ge $evalt_loss_thr ]
	then
	    g_evalt_loss_pass="True"
	fi
	##
	return 0
    fi
}

#evalt_program