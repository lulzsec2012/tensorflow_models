#!/bin/bash
trap "kill 0" INT

#####################################################
#                 Global Config
#####################################################
DATASET_DIR=/mllib/ImageNet/ILSVRC2012_tensorflow
DATASET_NAME=imagenet
TRAIN_DIR_PREFIX=./train_dir


#####################################################
#           Pruning and Retrain Config
#####################################################
MODEL_NAME=vgg_16
LABELS_OFFSET=1 #vgg resnet 1000+1 (1 for background)
checkpoint_path=./VGG_16_RETRAIN_FOR_CONVERGENCE_SGD_20000/model.ckpt-20000
DEFAULT_MAX_NUMBER_OF_STEPS=120
#layer_name pruning_rate max_number_of_steps
configs=(
    "vgg_16/fc8           0.23 $DEFAULT_MAX_NUMBER_OF_STEPS 20480"
    "vgg_16/fc7           0.04 $DEFAULT_MAX_NUMBER_OF_STEPS 16777216"
    "vgg_16/fc6           0.04 $DEFAULT_MAX_NUMBER_OF_STEPS 102760448"
    "vgg_16/conv5/conv5_3 0.36 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
    "vgg_16/conv5/conv5_2 0.29 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
    "vgg_16/conv5/conv5_1 0.35 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
    "vgg_16/conv4/conv4_3 0.34 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
    "vgg_16/conv4/conv4_2 0.27 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
    "vgg_16/conv4/conv4_1 0.32 $DEFAULT_MAX_NUMBER_OF_STEPS 1179648"
    "vgg_16/conv3/conv3_3 0.42 $DEFAULT_MAX_NUMBER_OF_STEPS 589824"
    "vgg_16/conv3/conv3_2 0.24 $DEFAULT_MAX_NUMBER_OF_STEPS 589824"
    "vgg_16/conv3/conv3_1 0.53 $DEFAULT_MAX_NUMBER_OF_STEPS 294912"
    "vgg_16/conv2/conv2_2 0.36 $DEFAULT_MAX_NUMBER_OF_STEPS 147456"
    "vgg_16/conv2/conv2_1 0.34 $DEFAULT_MAX_NUMBER_OF_STEPS 73728"
    "vgg_16/conv1/conv1_2 0.22 $DEFAULT_MAX_NUMBER_OF_STEPS 36864"
    "vgg_16/conv1/conv1_1 0.58 $DEFAULT_MAX_NUMBER_OF_STEPS 1728"
) 
Total_size_of_variables=134281029

##!!!checkpoint_exclude_scopes is the last layer_name of the array configs by default.
checkpoint_exclude_scopes=`echo "${configs[0]}" | awk  '{print $1}'`
echo "checkpoint_exclude_scopes:"$checkpoint_exclude_scopes


#####################################################
#              Algorithm Configs
#####################################################
#*Algorithm:Select_Pruning_Layers*#
#[...,bottom2,bottom1,currentLayer,top1,top2,...]
#PRUNING_LAYERS_BOTTOM=PRUNING_LAYERS_TOP=0 means pruning current layer only.
PRUNING_LAYERS_BOTTOM=0
PRUNING_LAYERS_TOP=0

#*Algorithm:REPRUNING_FROM_SPECIFIC_LAYER*#
En_REPRUNING_FROM_SPECIFIC_LAYER="Enable" #["Enable":"Disable"]
REPRUNING_FROM_LAYER_TH=0

#*Algorithm:AUTO_DRAW_BACK_WHILE_PRUNING*#


#*Algorithm:AUTO_RATE_PRUNING_WITHOUT_TRAINING*#
En_AUTO_RATE_PRUNING_WITHOUT_RETRAIN="Enable" #["Enable":"Disable"]


En_A_Retrain_for_ImageNet="Disable" #["Enable":"Disable"]


#####################################################
#              Pruning Functions
#####################################################
function parse_configs()
{
    local line="${configs[$1]}"
    local para=`echo $line | awk -v col="$2" '{print $col}'`
    echo $para
}

function get_multilayer_scopes()
{
    local row=$1
    local col=$2
    local comma_scopes=""
    for((iter=$row-$PRUNING_LAYERS_BOTTOM;iter<=$row+$PRUNING_LAYERS_TOP;iter++))
    do
	iter_scope=`parse_configs $iter $col`
	if [  $iter -ne -1  -a  -n "$iter_scope" ]
	then
	    comma_scopes="$comma_scopes,$iter_scope"
	fi
    done	
    scopes=${comma_scopes#,}
    echo "$scopes"
}

function next_CHECKPOINT_PATH()
{
    local train_dir=$1    
    if [ ! -f $train_dir/checkpoint ]
	then
	echo "Error:File $train_dir/checkpoint do not exit!"
	exit 1
    fi
    _ckpt_=`cat $train_dir/checkpoint | grep -v all_model_checkpoint_paths | awk '{print $2}'`
    ckpt_=${_ckpt_#\"}
    ckpt=${ckpt_%\"}
    checkpoint_path=$train_dir/$ckpt
    echo $checkpoint_path
}

function print_info()
{
    echo "######################################################"
    echo "pruning_and_retrain_step:" $*
    echo "CHECKPOINT_PATH:" $checkpoint_path "TRAIN_DIR:" $train_dir
    echo "######################################################"    
}

function pruning_and_retrain_step()
{
    #return 
    echo "Commands:" $@
    #global DATASET_DIR MODEL_NAME
    #make sure --max_number_of_steps=10 
    python train_image_classifier.py --noclone_on_cpu --optimizer sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	--max_number_of_steps=10 $@ --max_number_of_steps=10 
    python train_image_classifier.py --noclone_on_cpu --optimizer sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME $@
}

#####################################################
#               Eval Functions
#####################################################
function eval_image_classifier()
{
    # return "2017-08-15 20:34:18.682473: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.8275] 2017-08-15 20:34:18.682504: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]"
    local train_dir=$1
    local tmp_checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
    echo "train_dir="$train_dir
    echo "tmp_checkpoint_path="$tmp_checkpoint_path
    python eval_image_classifier.py --alsologtostderr --checkpoint_path=${tmp_checkpoint_path} --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=validation \
	--model_name=$MODEL_NAME --eval_dir ${train_dir} --labels_offset=$LABELS_OFFSET --max_num_batches=50 2>&1 | grep logging
#DATASET_DIR=/home/lzlu/work/tensorflow_models/inception/inception/data/train_directory ; CHECKPOINT_FILE=/tmp/flowers-models/inception_v3/model.ckpt-2689 ; python eval_image_classifier.py --alsologtostderr --checkpoint_path=${CHECKPOINT_FILE} --eval_dir eval_dir --dataset_dir=${DATASET_DIR} --dataset_name=imagenet --dataset_split_name=validation --model_name=vgg_16 --max_num_batches=50
}

function get_Accuracy()
{
    local result_str=$*
    local result=`echo $result_str | awk -F "Accuracy" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    echo $result_mul_10000
}

function get_Recall_5()
{
    local result_str=$*
    local result=`echo $result_str | awk -F "Recall_5" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    echo $result_mul_10000
}
# Usage:
#    result_str=`eval_image_classifier $train_dir`
#    Accuracy=`get_Accuracy $result_str` ; Recall_5=`get_Recall_5 $result_str`
#    echo "Accuracy =" $Accuracy "Recall_5 =" $Recall_5
#####################################################

function auto_rate_pruning()
{
    local pruning_step=10
    local allow_pruning_loss=50

    local begin_pruning_rate=50
    local count_step=4
    let "Accuracy_thr=preAccuracy-allow_pruning_loss"
    local local_cnt=0
    local count=0
    local layer_train_dir=$train_dir
    for((rate100=$begin_pruning_rate;rate100>0;rate100-=$pruning_step))
    do
	local rate="0.$rate100"
	echo -e "\n\n"
	rate_train_dir=${layer_train_dir}_$rate100
	rm $rate_train_dir -rf
	pruning_and_retrain_step $@ --pruning_rates_of_trainable_scopes=$rate \
            --checkpoint_path=${checkpoint_path}  --train_dir=${rate_train_dir}

	local result_str=`eval_image_classifier $rate_train_dir`
	local Accuracy=`get_Accuracy $result_str`
	local Recall_5=`get_Recall_5 $result_str`
	echo -e "Round "$count "Result:"
	echo "Accuracy="$Accuracy
	echo "preAccuracy="$preAccuracy
	echo "Accuracy_thr="$Accuracy_thr
	echo "local_cnt="$local_cnt
	echo "pruning_step="$pruning_step
	echo "rate="$rate

	if [ -z "$Accuracy" ]
	then
	    echo "Error!"
	    exit 0
	fi
	if [ "$Accuracy" -lt $Accuracy_thr ]
	then
	    let "rate100=rate100+pruning_step" #this is right
	    let "pruning_step=pruning_step/2"
	    let "local_cnt+=1"
	    echo "Draw Back $local_cnt Times."
	else
	    echo "Pass."
            if [ $rate100 -lt 10 ]
            then
                pruning_step=1
            fi
	fi
	if [ $local_cnt -ge $count_step -o $pruning_step -eq 0 ]
	then
	    let "rate100=rate100+pruning_step"
	    let "preAccuracy=Accuracy"
	    #output rate100
	    echo "preAccuracy="$preAccuracy #for next layer
	    break
	fi
	let "count+=1"
    done
    local rate="0.$rate100"
    echo "Finally the compress rate is:" $rate #output rate
}

#####################################################
#               Pruning Process Flow
#####################################################

#(A)Retrain for ImageNet 
train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Retrain_for_ImageNet
print_info "A"
pruning_and_retrain_step --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} \
    --learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64 --max_number_of_steps=600 
    #--checkpoint_exclude_scopes=$checkpoint_exclude_scopes --trainable_scopes=$checkpoint_exclude_scopes \
checkpoint_path=`next_CHECKPOINT_PATH $train_dir`

#Calculate and Print Eval Info
result_str=`eval_image_classifier $train_dir`
Accuracy=`get_Accuracy $result_str`
Recall_5=`get_Recall_5 $result_str`
preAccuracy=$Accuracy
echo "checkpoint_path =" $checkpoint_path
echo "preAccuracy =" $preAccuracy
echo "Recall_5 =" $Recall_5

#(B)Pruning without Retrain
if [ "$En_AUTO_RATE_PRUNING_WITHOUT_RETRAIN" = "Enable" ]
then
    g_train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Pruning_without_Retrain
    pruning_rates_without_retrain=""
    row=0
    for line in "${configs[@]}"
    do
	layer_name=`echo $line | awk '{print $1}'`
	pruning_rate=`echo $line | awk '{print $2}'`
	max_number_of_steps=`echo $line | awk '{print $3}'`
	
        #Pruning without Retrain
	print_info "B-$row" ; echo "Current config line --- configs[$row]:" $line

	train_dir=${g_train_dir}/$layer_name
	checkpoint_path=./VGG_16_RETRAIN_FOR_CONVERGENCE_SGD_20000/model.ckpt-20000 

	trainable_scopes=$layer_name
        pruning_scopes_pyramid="$pruning_scopes_pyramid,$layer_name"
	pruning_rates_pyramid="$pruning_rates_pyramid,$pruning_rate"
	
        ##Algorithm:AUTO_RATE_PRUNING_WITHOUT_TRAINING
	auto_rate_pruning --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} --max_number_of_steps=200 \
	    --trainable_scopes=$trainable_scopes --pruning_scopes=$trainable_scopes --pruning_rates_of_trainable_scopes=$pruning_rates_of_trainable_scopes --pruning_strategy=AUTO \
	    --learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64
	echo "checkpoint_path is:" $checkpoint_path
	let "row+=1"
    done
    exit 0
    pruning_rates_without_retrain=${pruning_rates_without_retrain#,} ## TODO:
    checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
fi
exit 0
#(C)Pruning and Retrain
trainable_scopes_pyramid=""
pruning_rates_of_trainable_scopes_pyramid=""
row=0
for line in "${configs[@]}"
do
    layer_name=`echo $line | awk '{print $1}'`
    pruning_rate=`echo $line | awk '{print $2}'`
    max_number_of_steps=`echo $line | awk '{print $3}'`
    trainable_scopes_pyramid="$trainable_scopes_pyramid,$layer_name"
    pruning_rates_of_trainable_scopes_pyramid="$pruning_rates_of_trainable_scopes_pyramid,$pruning_rate"

    #Pruning and Retrain
    train_dir=$TRAIN_DIR_PREFIX/$layer_name
    if [ $row -ge $REPRUNING_FROM_LAYER_TH -a "$En_REPRUNING_FROM_SPECIFIC_LAYER" = "Enable" ]
    then
	echo -e "\n\nRepruning $layer_name"
	rm $train_dir -rf #!!!!
    fi
    print_info "C-$row" ; echo "Current config line --- configs[$row]:" $line

    trainable_scopes=`get_multilayer_scopes $row 1`
    pruning_rates_of_trainable_scopes=`get_multilayer_scopes $row 2`

    pruning_and_retrain_step --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} --max_number_of_steps=$max_number_of_steps \
	--trainable_scopes=$trainable_scopes --pruning_scopes=$trainable_scopes --pruning_rates_of_trainable_scopes=$pruning_rates_of_trainable_scopes

    checkpoint_path=`next_CHECKPOINT_PATH $train_dir`

    let "row+=1"
done
trainable_scopes_pyramid=${trainable_scopes_pyramid#,}
pruning_rates_of_trainable_scopes_pyramid=${pruning_rates_of_trainable_scopes_pyramid#,}

#[D]Exit for Manual Modification
exit 0

#(E)Retrain the Prunned Network
train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Retrain_Prunned_Network
print_info "E"
pruning_and_retrain_step --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} \
    --trainable_scopes=$trainable_scopes_pyramid --pruning_scopes=$trainable_scopes_pyramid --pruning_rates_of_trainable_scopes=$pruning_rates_of_trainable_scopes_pyramid --max_number_of_steps=100
checkpoint_path=""
