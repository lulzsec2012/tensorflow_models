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
    for((step=10;step<$max_steps;step+=$evalt_interval))
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
    if [  $g_evalt_loss_pass = "True" ]
    then
	return 0
    else
	return 1
    fi
}

train_dir=./tmp/prune
checkpoint_path=./mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
check_dir=../mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
max_number_of_steps=200
all_trainable_scopes="LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1"
trainable_scopes=$all_trainable_scopes
pruning_scopes="LeNet/conv2,LeNet/conv1"
pruning_rates="0.8,0.8"
evalt_interval=50
g_early_skip="False"
pruning_layers_index="0 1"
#prune_and_evalt_step $train_dir $check_dir $max_number_of_steps $pruning_scopes $pruning_rates $trainable_scopes "$prune_net_args" $evalt_interval $g_early_skip
exit 0
#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step
#pruning_layers_index="0 1 2 3"
####################################################
function get_cur_iter() 
{
    local ckhp_iter_PC_txt=$1
    cur_iter=1
    if [ -f $ckhp_iter_PC_txt ]
    then
	cur_iter=`read_from_file $ckhp_iter_PC_txt`
	let "cur_iter+=1"
    fi
    echo "$cur_iter"
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

function get_iter_checkpoint_dir()
{
    local train_Dir=$1
    local checkpoint_path=$2
    local cur_iter=$3
    local all_trainable_scopes=$4
    let "pre_iter=cur_iter-1"
    local checkpoint_dir=${train_Dir}/iter${pre_iter}_pass 
    if [ $cur_iter -eq 1 ]
    then
	local ori_dir=`dirname $checkpoint_path`
	mkdir -p ${train_Dir}
	cp $ori_dir $checkpoint_dir  -rf
	
	local all_trainable_scopes_num=`echo $all_trainable_scopes | awk -F "," '{print NF}'`
	cur_rates="$g_starting_pruning_rate"
	cur_steps="$g_starting_pruning_step"
	for((i=1;i<$all_trainable_scopes_num;i+=1))
	do
	    cur_rates="$cur_rates,$g_starting_pruning_rate"    
	    cur_steps="$cur_steps,$g_starting_pruning_step"    
	done
	write_to_file ${checkpoint_dir}/iter_Rates.txt $cur_rates 2>&1 >> /dev/null
	write_to_file ${checkpoint_dir}/iter_Steps.txt $cur_steps 2>&1 >> /dev/null
    fi
    echo "$checkpoint_dir"
}

function is_substr()
{
    string=$1
    substr=$2

    local elem_num=`echo $string | awk -F "," '{print NF}'`
    local ii=0
    for((ii=1;ii<=$elem_num;ii+=1))
    do
	local elem=`echo $string | awk -v ith=$ii -F "," '{print $ith}'`
	if [ $elem -eq $substr ]
	then
	    echo "is_substr:elem="$elem
	    echo "is_substr:string="$string
	    return 0
	fi
    done
    return 1
}

function pruning_and_retrain_multilayers_iter()
{
    local all_trainable_scopes=$1 
    local iter=$2
    local allow_pruning_loss=$3 #0.2%*100
    local train_Dir=$4
    local checkpoint_Path=$5
    local pruning_layers_index=$6
    local pruning_singlelayer_retrain_step=$7
    local pruning_multilayers_retrain_step=$8
    En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable"
  
    let "g_Accuracy_thr=g_preAccuracy-allow_pruning_loss"
    let "g_Recall_5_thr=g_preRecall_5-allow_pruning_loss"

    ckhp_iter_PC_txt=${train_Dir}/ckhp_iter_PC.txt
    if [ $iter -eq -1 -o $iter -eq 0 ]
    then
	iter=`get_cur_iter $ckhp_iter_PC_txt`  ##auto get the current iter ; if not, use $2 instead.
    fi


    local checkpoint_dir=`get_iter_checkpoint_dir $train_Dir $checkpoint_Path $iter $all_trainable_scopes`
    local checkpoint_path=`next_CHECKPOINT_PATH $checkpoint_dir`

    local train_dir=${train_Dir}/iter$iter
    rm $train_dir/* -rf
    mkdir -p $train_dir
    checkpoint_dir=`dirname $checkpoint_path`
    cp ${checkpoint_dir}/iter_Rates.txt $train_dir
    cp ${checkpoint_dir}/iter_Steps.txt $train_dir
    ckhp_iter_Rates_txt="${train_dir}/iter_Rates.txt"
    ckhp_iter_Steps_txt="${train_dir}/iter_Steps.txt"

    echo "##############################"
    echo "multilayers_iter:iter="$iter
    echo "g_Accuracy_thr=$g_Accuracy_thr" ##
    En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable"

    local pruning_layers_num=`echo $pruning_layers_index | awk -F " " '{print NF}'`
    local count_stepzero=0
    for col in $pruning_layers_index
    do
	pruning_steps=`read_from_file $ckhp_iter_Steps_txt`	
	pruning_step=`get_str $pruning_steps $col`
	smaller=`awk -v numa=0.01 -v numb=$pruning_step 'BEGIN{print(numa>numb)?"1":"0"}'`
	if [ $smaller -eq 1 ]
	then
	    let "count_stepzero+=1"
	fi
    done
    echo "count_stepzero="$count_stepzero "pruning_layers_num="$pruning_layers_num
    if [ $count_stepzero -ge $pruning_layers_num ] #check if all pruning_steps all zero.
    then
	echo "Nothing can be done!"
	return 0
    fi
    
    local cnt=0
    local count_stepzero=0
    local count_preTry_Pass=0
    local pruning_layers_num=`echo $pruning_layers_index | awk -F " " '{print NF}'`
    is_preTry_Pass="True"
    for col in $pruning_layers_index
    do
	echo -e "\n\nmultilayers_iter:col=$col"
	echo "train_dir="$train_dir
	echo "checkpoint_path="$checkpoint_path
	echo "all_trainable_scopes="$all_trainable_scopes
	
	echo "pruning_layers_index="$pruning_layers_index

	pruning_rates=`read_from_file $ckhp_iter_Rates_txt`
	echo "pruning_rates(passed)=$pruning_rates"
	pruning_steps=`read_from_file $ckhp_iter_Steps_txt`
	pruning_rate=`get_str $pruning_rates $col`
	pruning_step=`get_str $pruning_steps $col`
	pruning_rate=`echo "scale=2;$pruning_rate-$pruning_step/1.0"|bc`
	echo "pruning_rate=$pruning_rate"
	if [ $is_preTry_Pass = "True" ]
	then
	    pruning_rates=`modify_str $pruning_rates $col  $pruning_rate` 
	fi
	echo "pruning_steps=$pruning_steps"

	local max_number_of_steps=`get_cur_number_of_steps $train_dir`
	echo "get_cur_number_of_steps:max_number_of_steps="$max_number_of_steps
	let "max_number_of_steps+=pruning_singlelayer_retrain_step"

	let "cnt+=1"
	if [ $pruning_layers_num -eq $cnt ]
	then
	    En_AUTO_RATE_PRUNING_EARLY_SKIP="Disable"
	    let "max_number_of_steps+=pruning_multilayers_retrain_step"
	fi
	##
	echo "train_dir="$train_dir
	echo "checkpoint_path="$checkpoint_path
	echo "ls $train_dir :"
	no_dot_dir=${train_dir#*"./"}
	train_dir_disk=${no_dot_dir#*/}
	echo "ROOT DIR OF THE PROJECT:${no_dot_dir%%/*}"
	
	ls `dirname $train_dir`

	echo "ls $train_dir ; grep iter pass -v used"
	ls $train_dir/../ | grep iter | grep -v `basename $train_dir` | grep -v `basename $checkpoint_dir`
	checkpoint_dir=`dirname $checkpoint_path`
	if [ ${no_dot_dir%%/*} = "train_dir_shm" ]
	then
	    ls $train_dir/../ | grep iter | grep -v `basename $train_dir` | grep -v `basename $checkpoint_dir` | xargs -i mv {} ./${train_dir_disk}/ -rf
	fi
	
	echo "ls $train_dir ; grep used"
	ls `dirname $train_dir` | grep  `basename $train_dir`
	ls `dirname $train_dir` | grep  `basename $checkpoint_dir` 
	##


	pruning_and_retrain_step_eval_multiLayer --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} \
            --trainable_scopes=$all_trainable_scopes --pruning_scopes=$all_trainable_scopes \
            --pruning_rates=$pruning_rates --max_number_of_steps=$max_number_of_steps --pruning_strategy=ABS \
	    --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 \
            --learning_rate=0.001  --weight_decay=0.0005 --batch_size=64  #2>&1 >> /dev/null
	if [ $? -ne 0 ]
	then
	    is_preTry_Pass="False"
	    En_AUTO_RATE_PRUNING_EARLY_SKIP="Disable"
	    pruning_rate=`echo "scale=2;$pruning_rate+$pruning_step/1.0"|bc`
	    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
	    rm ${train_dir} -rf
	    checkpoint_dir=`dirname $checkpoint_path`
	    cp $checkpoint_dir ${train_dir} -rf
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	    cp $ckhp_iter_Steps_txt $checkpoint_dir -rf
	else
	    is_preTry_Pass="True"
	    En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable"
	    let "count_preTry_Pass+=1"
	    write_to_file $ckhp_iter_Rates_txt "$pruning_rates"
	    smaller=`awk -v numa=$pruning_rate -v numb=$pruning_step 'BEGIN{print(numa-numb<0.001)?"1":"0"}'`
            if [ $smaller -eq 1 ]
            then
		pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
            fi
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	    rm ${train_dir}_pass -rf
	    cp ${train_dir} ${train_dir}_pass -rf
	    checkpoint_path=`next_CHECKPOINT_PATH ${train_dir}_pass`
	    write_to_file $ckhp_iter_PC_txt $iter
	fi    
	echo "is_preTry_Pass="$is_preTry_Pass
	echo "train_dir/checkpoint:"$train_dir/checkpoint
    done 
    if [ $count_stepzero -lt $pruning_layers_num ] #check if all pruning_steps all zero. #useless
    then 
	if [ ! -d ${train_dir}_pass ]
	then
            let "pre_iter=iter-1"
            echo "pre_iter:"$pre_iter
            #cp ${train_Dir}/iter${pre_iter}_pass ${train_dir}_pass -rfv ####
            mv ${train_Dir}/iter${pre_iter}_pass ${train_dir}_pass -fv ####
            cp $ckhp_iter_Rates_txt ${train_dir}_pass -rf
            cp $ckhp_iter_Steps_txt ${train_dir}_pass -rf
	fi
	rm ${train_dir} -rf
	write_to_file $ckhp_iter_PC_txt $iter
    fi
}

####################################################
function prune_and_evalt_scope()
{
    local train_dir=$1
    local check_dir=$2
    local max_steps=$3
    #local pruning_scopes=$4
    #local pruning_rates=$5
    local trainable_scopes=$4
    local prune_net_args=$5
    local evalt_interval=$6
    local g_early_skip=$7
    local pruning_layers_index=$8

    local pruning_layers_num=`echo $pruning_layers_index | awk -F " " '{print NF}'`
    local count_stepzero=0
    for col in $pruning_layers_index
    do
	pruning_rates=`read_from_file $ckhp_iter_Rates_txt`
	echo "pruning_rates(passed)=$pruning_rates"
	pruning_steps=`read_from_file $ckhp_iter_Steps_txt`
	pruning_rate=`get_str $pruning_rates $col`
	pruning_step=`get_str $pruning_steps $col`
	pruning_rate=`echo "scale=2;$pruning_rate-$pruning_step/1.0"|bc`
	smaller=`awk -v numa=0.01 -v numb=$pruning_step 'BEGIN{print(numa>numb)?"1":"0"}'`
	if [ $smaller -eq 1 ]
	then
	    let "count_stepzero+=1"
	fi
	prune_and_evalt_step $1 $2 $3 $pruning_scopes $4 $5 $6 $7 
    done
}

prune_and_evalt_scope $train_dir $check_dir $max_number_of_steps $trainable_scopes "$prune_net_args" $evalt_interval $g_early_skip $pruning_layers_index
#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step

function prune_and_evalt_iter()
{
    complete_iter=`get_complete_iter`
    for((iter=$iter_;iter<$max_iter;iter+=1))
    do
	prune_and_evalt_scope
	
    done
}

prune_net_args="" #all_trainable_scopes
evalt_net_args="" #all_trainable_scopes

