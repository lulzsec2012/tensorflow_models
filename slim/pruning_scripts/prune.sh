#!/bin/bash
trap "kill 0" INT
echo $CUDA_VISIBLE_DEVICES

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

prune_net_args="--noclone_on_cpu --optimizer=sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	--save_summaries_secs=$SAVE_SUMMARIES_SECS --pruning_gradient_update_ratio=0  --pruning_strategy=ABS \
        --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.001 --weight_decay=0.0005 --batch_size=64 \
         --num_clones=$NUM_CLONES" #all_trainable_scopes

evalt_net_args="--alsologtostderr  --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50" #all_trainable_scopes

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
    checkpoint_path=$train_dir/$ckpt
    echo $checkpoint_path
}

function train_program()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
    local prune_net_args=$7
    local tmp_check_path=`next_CHECKPOINT_PATH $check_dir`
    local cmd_str="--train_dir=$train_dir --checkpoint_path=$tmp_check_path --max_number_of_steps=$max_number_of_steps\
             --pruning_scopes=$pruning_scopes --pruning_rates=$pruning_rates --trainable_scopes=$trainable_scopes"
    echo $cmd_str $prune_net_args
    
    python3 ../train_image_classifier.py  $cmd_str $prune_net_args
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

#train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_rates train_scopes 
function prune_net()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
    local prune_net_args=$7
    echo "prune_net..."
    for((i=0;i<3;i+=1))
    do
	local nvidia_avaiable="True"
	if [ $nvidia_avaiable = "True" ]
	then
	    train_program "$@"
	    if [ $? -eq 0 ]
	    then
		return 0
	    fi
	fi
	sleep 60s
    done
    return 1
}

#train_dir evalt_loss_anc evalt_loss_thr evalt_loss_drp 
#g_evalt_loss_pass
function evalt_net() 
{
    echo "evalt_net..."
    for((i=0;i<3;i+=1))
    do
	local nvidia_avaiable="True"
	if [ $nvidia_avaiable = "True" ]
	then
	    evalt_program "$@"
	    if [ $? -eq 0 ]
	    then
		return 0
	    fi
	fi
	sleep 60s
    done
    return 1
}

#train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_rates train_scopes 

#Real eval command: python3 train_image_classifier.py --noclone_on_cpu --optimizer=sgd --labels_offset=1 --dataset_dir=/mllib/ImageNet/ILSVRC2012_tensorflow             --dataset_name=imagenet --dataset_split_name=train --model_name=vgg_16 	    --save_summaries_secs=1000 --checkpoint_path=./train_dir_multiLayers_imagenet_from_50000_learning_rate0.0001_reconfigGAccuracy_allowBiasBp_worker_replicas2_test_num_clones1_vgg_16/Retrain_Prunned_Network/iter0_pass/model.ckpt-50000 --train_dir=./train_dir_multiLayers_imagenet_from_50000_learning_rate0.0001_reconfigGAccuracy_allowBiasBp_worker_replicas2_test_num_clones1_vgg_16/Retrain_Prunned_Network/iter1 --trainable_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5/conv5_3,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_1,vgg_16/conv4/conv4_3,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_1,vgg_16/conv3/conv3_3,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_1,vgg_16/conv2/conv2_2,vgg_16/conv2/conv2_1,vgg_16/conv1/conv1_2,vgg_16/conv1/conv1_1 --pruning_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5/conv5_3,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_1,vgg_16/conv4/conv4_3,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_1,vgg_16/conv3/conv3_3,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_1,vgg_16/conv2/conv2_2,vgg_16/conv2/conv2_1,vgg_16/conv1/conv1_2,vgg_16/conv1/conv1_1 --pruning_rates=.96,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00 --max_number_of_steps=100010 --pruning_strategy=ABS --log_every_n_steps=10 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.0001 --weight_decay=0.0005 --batch_size=64 	    --max_number_of_steps=2000 --pruning_gradient_update_ratio=0 --num_clones=1

#train_dir check_dir  max_steps 
#max_steps evalt_interval  g_early_skip
function prune_and_evalt_step()
{
    local train_dir=$1
    local check_dir=$2
    local max_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
    local prune_net_args=$7
    local evalt_interval=$8
    local g_early_skip=$9

    local max_number_of_steps=0
    for((step=10;step<=$max_steps;step+=$evalt_interval))
    do
	echo "prune_and_evalt_step::step="$step "evalt_interval="$evalt_interval
	prune_net $1 $2 $step $4 $5 $6 "$prune_net_args" 
	evalt_net $1
	if [ $g_evalt_loss_pass = "True" -a $g_early_skip = "True" -a -d $train_dir ]
	then
	    break
	fi
	if [ $step -eq 10 -a $evalt_interval -gt 10 ]
	then
	    step=0
	fi
    done
    if [ $g_evalt_loss_pass = "True" ]
    then
	return 0
    else
	return 1
    fi
}


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
    mkdir -p $data_dir
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
    mkdir -p $data_dir
    for file in `\ls $data_dir | grep "iter${iter_major}"`
    do
	num=`echo $file | awk -F "-" '{print $2}'`
	_Max=`max $_Max $num`
	#echo $file $num  $_Max  
    done 
    echo "$_Max"
}


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

function get_str()
{
    local str=$1     
    local index=$2
    let "index=index+1"
    substr=`echo "$str" | awk -v col=$index -F "," '{print $col}'`
    echo "$substr"
}

function modify_str()
{
    local str=$1    
    local index=$2
    let "index=index+1"
    local substr=$3 
                    
    local NF=`echo "$str" | awk -F "," '{print NF}'`
    local _rates=""
    for((i=1;i<=$NF;i+=1))
    do
	cursubstr=`echo "$str" | awk -v col=$i -F "," '{print $col}'`
	if [ $i -eq $index ]
	then
	    _rates=$_rates,$substr
	else
	    _rates=$_rates,$cursubstr
	fi
    done
    _rates=${_rates#,}
    echo "$_rates"
}

function get_cur_number_of_steps()
{
    local train_dir=$1
    cur_number_of_steps=0
    if [ -f $train_dir/checkpoint ]
    then
	cur_number_of_steps=`cat $train_dir/checkpoint | grep -v "all_model_checkpoint_paths" | awk -F "-|\"" '{print $3}'`
    fi
    echo "$cur_number_of_steps"
}
#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step
#pruning_layers_index="0 1 2 3"
function prune_and_evalt_scope()
{
    local train_dir=$1
    local check_dir=$2
    local max_steps=$3
    local trainable_scopes=$4
    local prune_net_args=$5
    local evalt_interval=$6
    local g_early_skip=$7
    local pruning_layers_index=$8
    local pruning_singlelayer_prtrain_step=$9 
    local pruning_singlelayer_retrain_step=${10}
    local pruning_multilayers_retrain_step=${11}
    fdata_dir=./result
    iter_major=`get_current_iter_major $fdata_dir`
    iter_minor=`get_current_iter_minor $fdata_dir $iter_major`
    echo "the number of params:$#"
    echo "params:$@"
    echo "iter_major="$iter_major
    echo "iter_minor="$iter_minor
    echo "pruning_singlelayer_prtrain_step="$pruning_singlelayer_prtrain_step
    if [ $iter_major -eq 0 -a $iter_minor -eq 100 ]
    then
	mkdir -p ${fdata_dir}/iter0-100
	cp ${check_dir}/* ${fdata_dir}/iter0-100 -r
	check_dir=${fdata_dir}/iter0-100
    else
	check_dir=${fdata_dir}/iter${iter_major}-${iter_minor}_pass
    fi
    mkdir -p $train_dir
    cp ${check_dir}/iter_Rates.txt $train_dir
    cp ${check_dir}/iter_Steps.txt $train_dir
    ckhp_iter_Rates_txt="${train_dir}/iter_Rates.txt"
    ckhp_iter_Steps_txt="${train_dir}/iter_Steps.txt"
    cur_steps="$g_starting_pruning_step"
    cur_rates="$g_starting_pruning_rate"
    local all_trainable_scopes_num=`echo $all_trainable_scopes | awk -F "," '{print NF}'`
    for((i=1;i<$all_trainable_scopes_num;i+=1))
    do
	cur_steps="$cur_steps,$g_starting_pruning_step"    
	cur_rates="$cur_rates,$g_starting_pruning_rate"    
    done
    write_to_file $ckhp_iter_Rates_txt $cur_rates 2>&1 >> /dev/null
    write_to_file $ckhp_iter_Steps_txt $cur_steps 2>&1 >> /dev/null


    local count_stepzero=0
    for col in $pruning_layers_index
    do
	pruning_rates=`read_from_file $ckhp_iter_Rates_txt`
	echo "pruning_rates(passed)=$pruning_rates"
	pruning_steps=`read_from_file $ckhp_iter_Steps_txt`
	pruning_rate=`get_str $pruning_rates $col`
	pruning_step=`get_str $pruning_steps $col`
	pruning_rate=`echo "scale=2;$pruning_rate-$pruning_step/1.0"|bc`
	echo "pruning_rate=$pruning_rate"
	if [ $pruning_singlelayer_prtrain_step -gt 0 ]
	then
	    echo "prtrain: $1 $check_dir $pruning_singlelayer_prtrain_step $pruning_scopes $pruning_rates $4 "$5" $6 $7 "
	    prune_and_evalt_step $1 $check_dir $pruning_singlelayer_prtrain_step $pruning_scopes $pruning_rates $4 "$5" $6 $7 
	    rm $check_dir -rfv 
	    mv $train_dir $check_dir -v
	fi
	is_preTry_Pass="True"
	if [ $is_preTry_Pass = "True" ]
	then
	    pruning_rates=`modify_str $pruning_rates $col  $pruning_rate` 
	    echo "pruning_rates(modified)=$pruning_rates"
	fi
	echo "pruning_steps=$pruning_steps"

	local max_number_of_steps=`get_cur_number_of_steps $train_dir`
	echo "get_cur_number_of_steps:max_number_of_steps="$max_number_of_steps
	let "max_number_of_steps+=pruning_singlelayer_retrain_step"

	prune_and_evalt_step $1 $check_dir $pruning_singlelayer_retrain_step $pruning_scopes $pruning_rates $4 "$5" $6 $7 
	if [ $? -eq 0 ]
	then
	    echo "prune_and_evalt_step pass"
	    #TODO:update rates and steps
	    let "count_preTry_Pass+=1"
	    write_to_file $ckhp_iter_Rates_txt "$pruning_rates"
	    smaller=`awk -v numa=$pruning_rate -v numb=$pruning_step 'BEGIN{print(numa-numb<0.001)?"1":"0"}'`
            if [ $smaller -eq 1 ]
            then
		pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
            fi
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	    rm $check_dir -rf 
	    mv $train_dir $check_dir 
	else
	    echo "prune_and_evalt_step fail"
	    rm $check_dir -rf 
	    #TODO:update rates and steps
	    pruning_rate=`echo "scale=2;$pruning_rate+$pruning_step/1.0"|bc`
	    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
	    rm ${train_dir} -rf
	    mkdir -p ${train_dir}
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	fi
	exit 0
    done
}
train_dir=./tmp/prune
checkpoint_path=./mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
check_dir=../mnist_Train_from_Scratch_lenet/Retrain_from_Scratch
max_number_of_steps=200
all_trainable_scopes="LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1"
trainable_scopes=$all_trainable_scopes
pruning_scopes="LeNet/conv2,LeNet/conv1"
pruning_rates="0.8,0.8"
evalt_interval=50
g_early_skip="False"
pruning_layers_index="0 1"
#prune_and_evalt_step $train_dir $check_dir $max_number_of_steps $pruning_scopes $pruning_rates $trainable_scopes "$prune_net_args" $evalt_interval $g_early_skip
#exit 0
prune_and_evalt_scope $train_dir $check_dir $max_number_of_steps $trainable_scopes "$prune_net_args" $evalt_interval $g_early_skip "$pruning_layers_index" 0 100 150
exit 0
#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step

function prune_and_evalt_iter()
{
    iter_major=`get_current_iter_major`
    let "iter_count+=1"
    if [ $iter_count -gt $iter_major ]
    then
	return 0
    fi
    for((iter=0;iter<$max_iter;iter+=1))
    do
	prune_and_evalt_scope
	
    done
}

prune_net_args="" #all_trainable_scopes
evalt_net_args="" #all_trainable_scopes

