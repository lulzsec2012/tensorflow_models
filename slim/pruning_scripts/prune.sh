#!/bin/bash
trap "kill 0" INT
shopt -s  extglob
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

prune_net_args="--noclone_on_cpu --optimizer=sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	--save_summaries_secs=$SAVE_SUMMARIES_SECS --pruning_gradient_update_ratio=0  --pruning_strategy=ABS \
        --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.001 --weight_decay=0.0005 --batch_size=64 \
         --num_clones=$NUM_CLONES" #all_trainable_scopes

evalt_net_args="--alsologtostderr  --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50" #all_trainable_scopes
evalt_loss_anc=9990
evalt_loss_drp=50


#####################################################
#              Algorithm Configs
#####################################################


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

function train_program()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6

    local tmp_check_path=`next_CHECKPOINT_PATH $check_dir`
    local cmd_str="--train_dir=$train_dir --checkpoint_path=$tmp_check_path --max_number_of_steps=$max_number_of_steps\
             --pruning_scopes=$pruning_scopes --pruning_rates=$pruning_rates --trainable_scopes=$trainable_scopes"
    echo "python3 ../train_image_classifier.py  $prune_net_args $cmd_str"
    python3 ../train_image_classifier.py  $prune_net_args $cmd_str 
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

function prune_net()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
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

#train_dir check_dir  max_number_of_steps 
#max_number_of_steps evalt_interval  g_early_skip
function prune_and_evalt_step()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3 
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6

    local evalt_interval=$7
    local g_early_skip=$8

    local cur_step=`get_cur_number_of_steps $train_dir`
    local starting_step=10
    if [ $cur_step -ne 0 ]
    then
	if [ $g_early_skip = "True" ]
	then
	    let "starting_step+=cur_step"
	else
	    let "starting_step=evalt_interval"
	fi
	let "max_number_of_steps+=cur_step"
    fi
    echo "cur_step="$cur_step
    echo "max_number_of_steps="$max_number_of_steps
    echo "starting_step="$starting_step

    for((step=$starting_step;step<=$max_number_of_steps;step+=$evalt_interval))
    do
	echo "prune_and_evalt_step::step="$step "evalt_interval="$evalt_interval
	prune_net $1 $2 $step $4 $5 $6 
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
    local work_dir=${2:-work}
    local _Max=0
    mkdir -p $data_dir $work_dir
    for file in `\ls $data_dir $work_dir | grep iter | grep -v tmp | grep "-"`
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
    local work_dir=${2:-work}
    local iter_major=$3
    local _Max=100
    mkdir -p $data_dir $work_dir
    for file in `\ls $data_dir $work_dir | grep -v tmp | grep "iter${iter_major}"`
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


function prune_and_evalt_scope()
{
    local work_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local fdata_dir=$4
    local trainable_scopes=$5
    local evalt_interval=$6
    local g_early_skip=$7
    local pruning_singlelayer_prtrain_step=$8
    local pruning_singlelayer_retrain_step=${9}
    local pruning_multilayers_retrain_step=${10}
    local pruning_layers_index=${11}

    fdata_dir=./result
    train_dir=$work_dir/train_dir

    iter_major=`get_current_iter_major $fdata_dir $work_dir`
    iter_minor=`get_current_iter_minor $fdata_dir $work_dir $iter_major`
    echo "the number of params:$#"
    echo "params:$*"
    echo "iter_major="$iter_major
    echo "iter_minor="$iter_minor
    echo "pruning_singlelayer_prtrain_step="$pruning_singlelayer_prtrain_step
    if [ $iter_major -eq 0 -a $iter_minor -eq 100 ]
    then
	mkdir -p $fdata_dir/iter0_000
	cp ${check_dir}/* $fdata_dir/iter0_000 -r
	check_dir=$fdata_dir/iter0_000

	evalt_loss_anc=
	
	ckhp_iter_Rates_txt="${check_dir}/iter_Rates.txt"
	ckhp_iter_Steps_txt="${check_dir}/iter_Steps.txt"
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
    else
	if [ -d $work_dir/iter${iter_major}-${iter_minor} ]
	then
	    check_dir=$work_dir/iter${iter_major}-${iter_minor}
	else
	    check_dir=$fdata_dir/iter${iter_major}-${iter_minor}
	fi
    fi
    rm $train_dir -rf
    mkdir -p $train_dir
    cp ${check_dir}/iter_Rates.txt $train_dir
    cp ${check_dir}/iter_Steps.txt $train_dir
    ckhp_iter_Rates_txt="${train_dir}/iter_Rates.txt"
    ckhp_iter_Steps_txt="${train_dir}/iter_Steps.txt"


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
	if [ $pruning_singlelayer_prtrain_step -gt 0 -a 1 -eq 0 ]
	then
	    echo "prtrain: prune_and_evalt_step $train_dir $check_dir $pruning_singlelayer_prtrain_step $pruning_scopes $pruning_rates $trainable_scopes $evalt_interval $g_early_skip"
	    prune_and_evalt_step $train_dir $check_dir $pruning_singlelayer_prtrain_step $pruning_scopes $pruning_rates $trainable_scopes $evalt_interval $g_early_skip
	    rm $check_dir -rfv 
	    mv $train_dir $check_dir -vf
	fi
	is_preTry_Pass="True"
	if [ $is_preTry_Pass = "True" ]
	then
	    pruning_rates=`modify_str $pruning_rates $col  $pruning_rate` 
	    echo "pruning_rates(modified)=$pruning_rates"
	fi
	echo "pruning_steps=$pruning_steps"

	echo "envoke prune_and_evalt_step $train_dir $check_dir $pruning_singlelayer_retrain_step $pruning_scopes $pruning_rates $trainable_scopes $evalt_interval $g_early_skip"
       	prune_and_evalt_step $train_dir $check_dir $pruning_singlelayer_retrain_step $pruning_scopes $pruning_rates $trainable_scopes $evalt_interval $g_early_skip
	if [ $? -eq 0 ]
	then
	    echo "envoke prune_and_evalt_step pass"
	    let "count_preTry_Pass+=1"
	    write_to_file $ckhp_iter_Rates_txt "$pruning_rates"
	    smaller=`awk -v numa=$pruning_rate -v numb=$pruning_step 'BEGIN{print(numa-numb<0.001)?"1":"0"}'`
            if [ $smaller -eq 1 ]
            then
		pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
            fi
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	    rm 	$work_dir/check_dir -rf 
	    cp $train_dir $work_dir/check_dir -r
	    check_dir=$work_dir/check_dir
	else
	    echo "envoke prune_and_evalt_step fail"
	    pruning_rate=`echo "scale=2;$pruning_rate+$pruning_step/1.0"|bc`
	    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
	    if [ -d $work_dir/check_dir ]
	    then
		rm $train_dir -rf
		cp $check_dir  $train_dir -r
	    else
		echo "XXXXXXXXXXXXXXXXXXXXXXXXXX"
		echo "before rm:"
		ls $train_dir
		cd $train_dir
		rm -rf !(iter_Rates.txt)
		cd -
		echo "after rm:"
		ls $train_dir
	    fi
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	fi
    done
    let "iter_minor+=1"
    mkdir -p $check_dir
    rm $work_dir/iter${iter_major}-${iter_minor} -rf
    mv $work_dir/iter* $fdata_dir -vf
    mv $check_dir $work_dir/iter${iter_major}-${iter_minor} -vf
    ( rm $fdata_dir/*_tmp -rf ; cp $work_dir/iter${iter_major}-${iter_minor}  $fdata_dir/iter${iter_major}-${iter_minor}_tmp -rfv ) &
}

function prune_and_evalt_iter()
{
    local work_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3
    local fdata_dir=$4
    local trainable_scopes=$5
    local evalt_interval=$6
    local g_early_skip=$7
    local pruning_singlelayer_prtrain_step=$8
    local pruning_singlelayer_retrain_step=${9}
    local pruning_multilayers_retrain_step=${10}
    local max_minor_iter=${11}
    local pruning_layers_index=${12}

    iter_major=`get_current_iter_major $fdata_dir $work_dir`
    cur_iter_count=$ITER_COUNT
    let "ITER_COUNT+=1"
    
    if [ $iter_major -gt $cur_iter_count ]
    then
	return 0
    fi

    for((minor_iter=0;minor_iter<$max_minor_iter;minor_iter+=1))
    do
	echo "####################################################################"
	echo "# 'CURRENT_ITER='$cur_iter_count   'minor_iter='$minor_iter        #"
	echo "####################################################################"
	prune_and_evalt_scope $work_dir $check_dir $max_number_of_steps $fdata_dir $trainable_scopes $evalt_interval $g_early_skip $pruning_singlelayer_prtrain_step $pruning_singlelayer_retrain_step \
	    $pruning_multilayers_retrain_step "$pruning_layers_index"
    done
}


work_dir=./tmp/prune
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
fdata_dir=./result


ITER_COUNT=0 
prune_and_evalt_iter $work_dir $check_dir $max_number_of_steps $fdata_dir $trainable_scopes $evalt_interval $g_early_skip 0 100 150 2 "$pruning_layers_index"
prune_and_evalt_iter $work_dir $check_dir $max_number_of_steps $fdata_dir $trainable_scopes $evalt_interval $g_early_skip 0 100 150 2 "$pruning_layers_index"
prune_and_evalt_iter $work_dir $check_dir $max_number_of_steps $fdata_dir $trainable_scopes $evalt_interval $g_early_skip 0 100 150 2 "$pruning_layers_index"
