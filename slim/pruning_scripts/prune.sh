#!/bin/bash
shopt -s  extglob
shopt -s expand_aliases
alias log='tee -a $LOG_FILE'
#####################################################
#                 Global Config
#####################################################

############################Configs############################
MODEL_NAME=lenet
DATASET_NAME=mnist
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
elif [ $DATASET_NAME = "imagenet" ]
then
    LABELS_OFFSET=1  #vgg resnet 1000+1 (1 for background)	
    DATASET_DIR=/mllib/ImageNet/ILSVRC2012_tensorflow	
else	
    :
fi

#####################################################
#              Functions
#####################################################

function get_extra_args()
{
    array=( $@ )
    len=${#array[@]}
    start=$1
    let "start+=1"
    EXTRA_ARGS=${array[@]:$start:$len}
    EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
    echo "$EXTRA_ARGS"
}

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
    local extra_args=`get_extra_args 6 $@`

    local tmp_check_path=`next_CHECKPOINT_PATH $check_dir`
    local cmd_str="--train_dir=$train_dir --checkpoint_path=$tmp_check_path --max_number_of_steps=$max_number_of_steps\
             --pruning_scopes=$pruning_scopes --pruning_rates=$pruning_rates --trainable_scopes=$trainable_scopes"
    echo "python3 ../train_image_classifier.py  $prune_net_args $cmd_str" ${extra_args}
    python3 ../train_image_classifier.py  $prune_net_args $cmd_str ${extra_args}
}

function get_Recall_5()
{
    local result_str="$*" #`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Recall_5" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Recall_5() failed!"
	g_Recall_5=0
	return 1
    else
	g_Recall_5=$result_mul_10000
	return 0
    fi
}

function get_Accuracy()
{
    local result_str="$*" #`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Accuracy" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Accuracy() failed!"
	g_Accuracy=0
	return 1
    else
	g_Accuracy=$result_mul_10000
	return 0
    fi
}

#USE_EVALT_LOSS_PRE
USE_EVALT_LOSS_PRE="True"
evalt_loss_pre=9999
function evalt_program()
{
    local train_dir=$1
    local evalt_loss_anc=$2
    local evalt_loss_drp=$3
    local edata_dir=${4:-${train_dir}/eval_event}

    local tmp_checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
    cmd_str="--checkpoint_path=${tmp_checkpoint_path}  --eval_dir=$edata_dir"   
    echo "evalt_program: python3 ../eval_image_classifier.py " $cmd_str $evalt_net_args 
    local result_str=`python3 ../eval_image_classifier.py  $cmd_str $evalt_net_args   2>&1 | grep logging`
    get_Accuracy $result_str && echo "Info:get_Accuracy Pass!" || (echo "Error:get_Accuracy Fail!" ;exit 0) | log 
    get_Recall_5 $result_str && echo "Info:get_Recall_5 Pass!" || (echo "Error:get_Recall_5 Fail!" ;exit 0) | log
    echo "g_Accuracy="$g_Accuracy | log 
    echo "g_Recall_5="$g_Recall_5 | log

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
	echo "EVALT:" 
	echo "evalt_loss_anc="$evalt_loss_anc "evalt_loss_drp="$evalt_loss_drp | log
	echo "evalt_loss_cur="$evalt_loss_cur "evalt_loss_thr="$evalt_loss_thr | log
	if [ $evalt_loss_cur -ge $evalt_loss_thr ]
	then
	    g_evalt_loss_pass="True"
	fi
	##
	if [ $evalt_loss_cur -ge $evalt_loss_pre -a $evalt_loss_pre -ne 0 -a $USE_EVALT_LOSS_PRE = "True" ]
	then
	    g_evalt_loss_pass="True"
	    echo "USE_EVALT_LOSS_PRE=True"
	    echo "evalt_loss_pre="$evalt_loss_pre
	fi
	evalt_loss_pre=${evalt_loss_cur:-0}
	
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
    local extra_args=`get_extra_args 6 $@`

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
	    else
		cd $train_dir
		rm -rf !(iter_Rates.txt iter_Steps.txt)
		cd -
		let "ti=i+1"
		echo "Error: envoke train_program failed ${ti} times." 
	    fi
	fi
	sleep 60s
    done
    return 1
}

function evalt_net() 
{
    local train_dir=$1
    local evalt_loss_anc=$2
    local evalt_loss_drp=$3
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
	    else
		let "ti=i+1"
		echo "Error: envoke evalt_program failed ${ti} times." 
	    fi
	fi
	sleep 60s
    done
    return 1
}

function prune_and_evalt_step()
{
    local train_dir=$1
    local check_dir=$2
    local max_number_of_steps=$3 
    local pruning_scopes=$4
    local pruning_rates=$5
    local trainable_scopes=$6
    local evalt_loss_anc=$7
    local evalt_loss_drp=$8
    local evalt_interval=$9
    local early_skip=${10}
    local extra_args=`get_extra_args 10 $@`
    echo "prune_and_evalt_step:extra_args="$extra_args
    
    local cur_step=`get_cur_number_of_steps $train_dir`
    local starting_step=10
    if [ $cur_step -ne 0 ]
    then
	if [ $early_skip = "True" ]
	then
	    let "starting_step+=cur_step"
	    echo "A:starting_step="$starting_step
	else
	    let "starting_step=cur_step+evalt_interval"	    
	    echo "B:starting_step="$starting_step
	fi
	let "max_number_of_steps+=cur_step"
    fi
    echo "cur_step="$cur_step

    for((step=$starting_step;step<=$max_number_of_steps;step+=$evalt_interval))
    do
	echo -e "\n\nprune_and_evalt_step::starting_step="$starting_step "step="$step "evalt_interval="$evalt_interval "max_number_of_steps="$max_number_of_steps | log 
	prune_net $train_dir $check_dir $step $pruning_scopes $pruning_rates $trainable_scopes ${extra_args}
	if [ $? -eq 1 ]
	then
	    return 2
	fi
	evalt_net $train_dir $evalt_loss_anc $evalt_loss_drp
	if [ $g_evalt_loss_pass = "True" -a $early_skip = "True" -a -d $train_dir -a $step -ne 10 ]
	then
	    echo "g_evalt_loss_pass = True!" ###
	    break
	else
	    echo "g_evalt_loss_pass = False!" ###
	    echo $g_evalt_loss_pass $early_skip $step
	    ls $train_dir
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

function set_iter_steps_rates()
{
    local all_trainable_scopes=$1
    local train_dir=$2
    local g_starting_pruning_step=$3
    local g_starting_pruning_rate=$4
    local set_both_steps_and_rates=${5:-"True"}

    if [ ! -d $train_dir ]
    then
	mkdir -p $train_dir
    fi
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
    write_to_file $ckhp_iter_Steps_txt $cur_steps 2>&1 >> /dev/null
    if [ $set_both_steps_and_rates = "True" ]
    then
	write_to_file $ckhp_iter_Rates_txt $cur_rates 2>&1 >> /dev/null
    fi
}

function set_check_dir()
{
    iter_major=`get_current_iter_major $fdata_dir $work_dir`
    iter_minor=`get_current_iter_minor $fdata_dir $work_dir $iter_major`
    if [ $iter_major -eq 0 -a $iter_minor -eq 100 ]
    then
	mkdir -p $fdata_dir/iter0_000
	cp ${check_dir}/* $fdata_dir/iter0_000 -r
	check_dir=$fdata_dir/iter0_000
	set_iter_steps_rates $all_trainable_scopes $check_dir $g_starting_pruning_step $g_starting_pruning_rate "True"
    else
	if [ -d $work_dir/iter${iter_major}-${iter_minor} ]
	then
	    check_dir=$work_dir/iter${iter_major}-${iter_minor}
	else
	    check_dir=$fdata_dir/iter${iter_major}-${iter_minor}
	fi
    fi
    if [ ! -f $check_dir/iter_Rates.txt ]
    then
	rm $check_dir -rf
	echo "$check_dir is empty, get a new check_dir!"
	set_check_dir
    fi
}

function set_train_dir()
{
    train_dir=$work_dir/train_dir
    #rm $train_dir -rf #note this for prtrain.
    mkdir -p $train_dir
    cp ${check_dir}/iter_Rates.txt $train_dir
    cp ${check_dir}/iter_Steps.txt $train_dir
    ckhp_iter_Rates_txt="${train_dir}/iter_Rates.txt"
    ckhp_iter_Steps_txt="${train_dir}/iter_Steps.txt"
    if [ $iter_major -ne $iter_count ]
    then
	iter_major=$iter_count
	iter_minor=100
	set_iter_steps_rates $all_trainable_scopes $train_dir $pruning_rate_drop_step $g_starting_pruning_rate "False"
    fi
}

function check_all_pruning_steps_zero()
{
    local pruning_layers_num=`echo $pruning_layers_index | awk -F " " '{print NF}'`
    local count_stepzero=0
    if [ "$pruning_layers_index" = "-" ]
    then
	echo "check_all_pruning_steps_zero:pruning_layers_index="-""
	return 0
    fi
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
	return 1
    fi
    return 0
}

function prune_and_evalt_scope()
{
    local work_dir=$1
    local check_dir=$2
    local iter_count=$3
    local fdata_dir=$4
    local trainable_scopes=$5
    local evalt_loss_anc=$6
    local evalt_loss_drp=$7
    local pruning_rate_drop_step=$8
    local evalt_interval=$9
    local early_skip=${10}
    local pruning_singlelayer_prtrain_step=${11}
    local pruning_singlelayer_retrain_step=${12}
    local pruning_multilayers_retrain_step=${13}
    local pruning_layers_index=${14}
    local extra_args=`get_extra_args 14 $@`
    echo "prune_and_evalt_scope:extra_args="$extra_args

    iter_major=`get_current_iter_major $fdata_dir $work_dir`
    iter_minor=`get_current_iter_minor $fdata_dir $work_dir $iter_major`
    echo -e "\n\nXXXXXXXXXXXXXXXXXprune_and_evalt_scopeXXXXXXXXXXXXXXXXX"
    echo "the number of params:$#" 
    echo "params:$*"  | log  
    echo "iter_major="$iter_major
    echo "iter_minor="$iter_minor
    echo "pruning_singlelayer_prtrain_step="$pruning_singlelayer_prtrain_step

    set_check_dir
    set_train_dir


    local act_trainable_scopes=""
    echo "pruning_layers_index(,)="$pruning_layers_index
    first_col=`echo $pruning_layers_index | awk -F "," '{print $1}'`
    if [ $first_col = "-" ]
    then
	pruning_layers_index=${pruning_layers_index#*,}
    fi
    pruning_layers_index=${pruning_layers_index//,/ }
    echo "pruning_layers_index( )="$pruning_layers_index
    for _col in $pruning_layers_index
    do
	let "_col_add1=_col+1"
	scope=`echo $trainable_scopes | awk -F "," -v col=$_col_add1 '{print $col}'`
	act_trainable_scopes=$act_trainable_scopes,$scope
    done
    act_trainable_scopes=${act_trainable_scopes#,}
    echo "act_trainable_scopes="$act_trainable_scopes
    if [ $first_col = "-" ]
    then
	pruning_layers_index="-"
	echo "pruning_layers_index(final)="$pruning_layers_index
    fi    


    check_all_pruning_steps_zero
    if [ $? -eq 1 ]
    then
	return 0
    fi

    local cnt=0
    for col in $pruning_layers_index
    do
	echo -e "\n\nprune_and_evalt_scope:col="$col | log
	if [ ! -f $train_dir/iter_Rates.txt -o ! -f $train_dir/iter_Steps.txt ]
	then
	    set_train_dir
	fi
	read_from_file ${train_dir}/iter_Rates.txt | log  
	pruning_rates=`read_from_file ${train_dir}/iter_Rates.txt`
	pruning_steps=`read_from_file ${train_dir}/iter_Steps.txt`
	echo "pruning_layers_index=$pruning_layers_index" | log 
	echo "pruning_rates(passed)=$pruning_rates" | log 
	echo "pruning_rate=$pruning_rate"
	if [ $col != "-" ]
	then
	    pruning_rate=`get_str $pruning_rates $col`
	    pruning_step=`get_str $pruning_steps $col`
	    pruning_rate_pruning=`echo "scale=2;$pruning_rate-$pruning_step/1.0"|bc`
	    echo "pruning_rate_pruning="$pruning_rate_pruning
	    smaller=`awk -v numa=$pruning_rate_pruning -v numb=0.001 'BEGIN{print(numa-numb<0.001)?"1":"0"}'`
	    if [ $smaller -eq 1 ]
	    then
		:
	    else
		pruning_rate=$pruning_rate_pruning
	    fi
	    pruning_rates=`modify_str $pruning_rates $col  $pruning_rate` 
	fi
	echo "pruning_rates(modified)=$pruning_rates" | log 
	echo "pruning_steps=$pruning_steps" | log 

	if [ ! -f $check_dir/iter_Rates.txt ]
	then
	    rm $check_dir -rf
	    echo "$check_dir is empty, get a new check_dir!"
	    set_check_dir
	fi

	let "cnt+=1"
	local max_number_of_steps=0
	local pruning_layers_num=`echo $pruning_layers_index | awk -F "," '{print NF}'`
	if [ $pruning_layers_num -eq $cnt -a $pruning_multilayers_retrain_step -ne 0 ]
	then
	    echo "use pruning_multilayers_retrain_step!"
	    early_skip="False"
	    let "max_number_of_steps=pruning_multilayers_retrain_step+pruning_singlelayer_retrain_step"
	else
	    early_skip=$early_skip
	    max_number_of_steps=$pruning_singlelayer_retrain_step
	fi

	if [ $pruning_singlelayer_prtrain_step -gt 0 ]
	then
	    echo "pruning_singlelayer_prtrain_step="$pruning_singlelayer_prtrain_step | log
	    early_skip="False"
	    max_number_of_steps=$pruning_singlelayer_prtrain_step
	fi

	local pruning_step_zero=`awk -v numa=0.01 -v numb=$pruning_step 'BEGIN{print(numa>numb)?"1":"0"}'`
	echo "prtrain envoke prune_and_evalt_step $train_dir $check_dir $max_number_of_steps $trainable_scopes $pruning_rates $act_trainable_scopes $evalt_loss_anc $evalt_loss_drp $evalt_interval $early_skip" ${extra_args} | log 
       	prune_and_evalt_step $train_dir $check_dir $max_number_of_steps $trainable_scopes $pruning_rates $act_trainable_scopes $evalt_loss_anc $evalt_loss_drp $evalt_interval $early_skip ${extra_args}
	flag=$?
	echo $flag
	if [ $flag -eq 1 -a $pruning_step_zero -eq 1 -a $col != "-" ]
	then
	    break
	elif [ $flag -eq 0 -o $col = "-" ]
	then
	    echo "envoke prune_and_evalt_step PASS 0" | log 
	    let "count_preTry_Pass+=1"
	    write_to_file ${train_dir}/iter_Rates.txt "$pruning_rates"
	    if [ $col != "-" ]
	    then
		smaller=`awk -v numa=$pruning_rate -v numb=$pruning_step 'BEGIN{print(numa-numb<0.001)?"1":"0"}'`
		if [ $smaller -eq 1 ]
		then
		    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
		fi
		pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    fi
	    write_to_file ${train_dir}/iter_Steps.txt "$pruning_steps"
	    ls $work_dir $fdata_dir
	    rm 	$work_dir/check_dir -rf 
	    cp $train_dir $work_dir/check_dir -r
	    check_dir=$work_dir/check_dir
	    ls $work_dir $fdata_dir 
	elif [ $flag -eq 1 ]
	then
	    echo "envoke prune_and_evalt_step FAIL 1" | log
	    pruning_rate=`echo "scale=2;$pruning_rate+$pruning_step/1.0"|bc`
	    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
	    ls $work_dir $fdata_dir
	    if [ -d $work_dir/check_dir ]
	    then
		rm $train_dir -rf
		cp $check_dir  $train_dir -rv
	    else
		cd $train_dir
		rm -rf !(iter_Rates.txt)
		cd -
	    fi
	    ls $work_dir $fdata_dir 
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file ${train_dir}/iter_Steps.txt "$pruning_steps"
	    ##
	    ##evalt_net $train_dir $evalt_loss_anc $evalt_loss_drp
	    ##if [ $g_evalt_loss_pass = "True" ]
	    ##then
		##echo "g_evalt_loss_pass = True!" ###
		#break
	    ##else
		##echo "g_evalt_loss_pass = False!" ###
	    ##fi
	    ##
	else #[ $flag -eq 2 ]
	    echo "envoke prune_and_evalt_step ERROR 2" | log
	    rm $train_dir -rf
	    set_check_dir
	    set_train_dir	    
	fi
    done
    let "iter_minor+=1"
    mkdir -p $check_dir
    rm $work_dir/iter${iter_major}-${iter_minor} -rf
    ##( mv $work_dir/iter* $fdata_dir -f ) & ## 
    ##( cd $work_dir ; ls | grep iter | xargs -I {} mv {} $fdata_dir ) &
    cd $work_dir ; iter_file_list=`\ls | grep iter` ; cd -
    pwd_dir=$PWD
    echo "check if mv is wrong! pwd_dir="$pwd_dir
    ( cd $work_dir ; echo "$iter_file_list" | xargs -I {} mv {} $pwd_dir/$fdata_dir -v ) &
    if [ `find $train_dir -name "model*" | wc -l` -gt 0 ]
    then
	mv $train_dir $work_dir/iter${iter_major}-${iter_minor} -vf
    else
	rm $train_dir -rf
    fi
    ##( rm $fdata_dir/*_tmp -rf ; cp $work_dir/iter${iter_major}-${iter_minor}  $fdata_dir/iter${iter_major}-${iter_minor}_tmp -rfv ) &
}

function prune_and_evalt_iter()
{
    local work_dir=$1
    local check_dir=$2
    local max_iters=$3
    local fdata_dir=$4
    local trainable_scopes=$5
    local evalt_loss_anc=$6
    local evalt_loss_drp=$7
    local pruning_rate_drop_step=$8
    local evalt_interval=$9
    local early_skip=${10}
    local pruning_singlelayer_prtrain_step=${11}
    local pruning_singlelayer_retrain_step=${12}
    local pruning_multilayers_retrain_step=${13}
    local pruning_layers_index=${14}
    local extra_args=`get_extra_args 14 $@`
    echo "prune_and_evalt_iter:extra_args="$extra_args

    iter_major=`get_current_iter_major $fdata_dir $work_dir`
    cur_iter_count=$ITER_COUNT
    let "ITER_COUNT+=1"
    echo "iter_major="$iter_major "cur_iter_count="$cur_iter_count
    if [ $iter_major -gt $cur_iter_count ]
    then
	echo "iter done!"
	return 0
    fi

    local iter_minor=`get_current_iter_minor $fdata_dir $work_dir $cur_iter_count`
    let "iter_minor=iter_minor-100"
    for((minor_iter=$iter_minor;minor_iter<$max_iters;minor_iter+=1))
    do
	echo "####################################################################" | log
	echo "# 'CURRENT_ITER='$cur_iter_count   'minor_iter='$minor_iter        #" | log
	echo "####################################################################" | log
	date | log 
	prune_and_evalt_scope $work_dir $check_dir $cur_iter_count $fdata_dir $trainable_scopes $evalt_loss_anc $evalt_loss_drp $pruning_rate_drop_step $evalt_interval \
	    $early_skip $pruning_singlelayer_prtrain_step $pruning_singlelayer_retrain_step \
	    $pruning_multilayers_retrain_step "$pruning_layers_index" ${extra_args}
	date | log 
    done
}

#####################################################
#      Global Flags (may change dure the program)
#####################################################
g_starting_pruning_rate=1.00
g_starting_pruning_step=0.10

prune_net_args="--noclone_on_cpu --optimizer=sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	--save_summaries_secs=200 --pruning_gradient_update_ratio=0  --pruning_strategy=ABS \
        --log_every_n_steps=50 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.00001 --weight_decay=0.0005 --batch_size=64 \
         --num_clones=$NUM_CLONES" 

evalt_net_args="--alsologtostderr  --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL --model_name=$MODEL_NAME --max_num_batches=50 --labels_offset=$LABELS_OFFSET"
evalt_loss_anc=0
evalt_loss_drp=50

check_dir=../mnist_Train_from_Scratch_lenet/Retrain_from_Scratch
#check_dir=fashion_mnist_50000_0.01_sgd
#check_dir=/home/lzlu/tensorflow_models/slim/VGG_16_RETRAIN_FOR_CONVERGENCE_SGD_20000
all_trainable_scopes="LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1"
#all_trainable_scopes="vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5/conv5_3,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_1,vgg_16/conv4/conv4_3,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_1,vgg_16/conv3/conv3_3,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_1,vgg_16/conv2/conv2_2,vgg_16/conv2/conv2_1,vgg_16/conv1/conv1_2,vgg_16/conv1/conv1_1"

fdata_dir=result_lenet_mnist_230_CROSS_1020_0404_tmp_tt
work_dir=$shm_dir/$fdata_dir
max_iters=10
trainable_scopes=$all_trainable_scopes
evalt_interval=50
early_skip="True"
pruning_layers_index="0,1"

LOG_FILE=$fdata_dir/log

#####################################################
#             Starting From Here
#####################################################
mkdir -p $fdata_dir
cp $0 $fdata_dir -v
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
trap "(mv $work_dir/iter* $fdata_dir -vf ; echo 'mv OK' )&& rm $work_dir -rvf ; kill 0" INT

echo "Computing evalt_loss_anc:" | log 
evalt_net $check_dir 0 0 /tmp #2>&1 >> /dev/null
evalt_loss_anc=$g_Accuracy
echo "evalt_loss_anc="$evalt_loss_anc | log 

ITER_COUNT=0 
#prune_and_evalt_iter $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc $evalt_loss_drp $pruning_rate_drop_step $evalt_interval $early_skip  1000 100 150 $pruning_layers_index
#prune_and_evalt_iter  $work_dir $check_dir          1 $fdata_dir $trainable_scopes $evalt_loss_anc            100                    0.04              50 $early_skip  3000  100 0   "0"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.10  50 $early_skip  0  50 0   "0,1"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,0,1"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.10  50 $early_skip  0  50 0   "2,3"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04 500     "False"  0 500 0   "-,2,3"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "1,0"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,1,0"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "3,2"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,3,2"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "0,1"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,0,1"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "2,3"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,2,3"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "1,0"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,1,0"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "3,2"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,3,2"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "0,1"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,0,1"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04  50 $early_skip  0  50 0   "2,3"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   100  0.04 500     "False"  0 500 0   "-,2,3"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "1,0"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,1,0"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "3,2"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,3,2"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "0,1"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,0,1"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "2,3"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,2,3"       "--learning_rate=0.001"

prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "1,0"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,1,0"       "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04  50 $early_skip  0  50 0   "3,2"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc    50  0.04 500     "False"  0 500 0   "-,3,2"       "--learning_rate=0.001"

#prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04  50 $early_skip  0  50 0   "0,1"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04 500     "False"  0 500 0   "-,0,1"       "--learning_rate=0.0001"
#prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04  50 $early_skip  0  50 0   "2,3"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04 500     "False"  0 500 0   "-,2,3"       "--learning_rate=0.0001"

#prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04  50 $early_skip  0  50 0   "1,0"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04 500     "False"  0 500 0   "-,1,0"       "--learning_rate=0.00001"
#prune_and_evalt_iter  $work_dir $check_dir $max_iters $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04  50 $early_skip  0  50 0   "3,2"         "--learning_rate=0.001"
prune_and_evalt_iter  $work_dir $check_dir          6 $fdata_dir $trainable_scopes $evalt_loss_anc   200  0.04 500     "False"  0 500 0   "-,3,2"       "--learning_rate=0.00001"

mv $work_dir/iter* $fdata_dir -vf && rm $work_dir -rvf
