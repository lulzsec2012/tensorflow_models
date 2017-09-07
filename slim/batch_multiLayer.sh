#!/bin/bash
trap "kill 0" INT

#####################################################
#                 Global Config
#####################################################
DATASET_DIR=/tmp/mnist #/mllib/ImageNet/ILSVRC2012_tensorflow
DATASET_NAME=mnist     #imagenet
TRAIN_DIR_PREFIX=./train_dir_multiLayer
EVAL_INTERVAL=50
SAVE_SUMMARIES_SECS=50
DEFAULT_MAX_NUMBER_OF_STEPS=200
#DATASET_SPLIT_NAME_FOR_VAL=validation #for vgg
DATASET_SPLIT_NAME_FOR_VAL=test #for mnist
#####################################################
#           Pruning and Retrain Config
#####################################################
MODEL_NAME=lenet #vgg_16
LABELS_OFFSET=0  #vgg resnet 1000+1 (1 for background)
checkpoint_path= #./VGG_16_RETRAIN_FOR_CONVERGENCE_SGD_20000/model.ckpt-20000
#layer_name pruning_rate max_number_of_steps
configs_vgg=(
"vgg_16/conv1/conv1_1 0.58 $DEFAULT_MAX_NUMBER_OF_STEPS 1728"
"vgg_16/conv1/conv1_2 0.22 $DEFAULT_MAX_NUMBER_OF_STEPS 36864"
"vgg_16/conv2/conv2_1 0.34 $DEFAULT_MAX_NUMBER_OF_STEPS 73728"
"vgg_16/conv2/conv2_2 0.36 $DEFAULT_MAX_NUMBER_OF_STEPS 147456"
"vgg_16/conv3/conv3_1 0.53 $DEFAULT_MAX_NUMBER_OF_STEPS 294912"
"vgg_16/conv3/conv3_2 0.24 $DEFAULT_MAX_NUMBER_OF_STEPS 589824"
"vgg_16/conv3/conv3_3 0.42 $DEFAULT_MAX_NUMBER_OF_STEPS 589824"
"vgg_16/conv4/conv4_1 0.32 $DEFAULT_MAX_NUMBER_OF_STEPS 1179648"
"vgg_16/conv4/conv4_2 0.27 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
"vgg_16/conv4/conv4_3 0.34 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
"vgg_16/conv5/conv5_1 0.35 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
"vgg_16/conv5/conv5_2 0.29 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
"vgg_16/conv5/conv5_3 0.36 $DEFAULT_MAX_NUMBER_OF_STEPS 2359296"
"vgg_16/fc6           0.10 $DEFAULT_MAX_NUMBER_OF_STEPS 102760448"
"vgg_16/fc7           0.10 $DEFAULT_MAX_NUMBER_OF_STEPS 16777216"
"vgg_16/fc8           0.23 $DEFAULT_MAX_NUMBER_OF_STEPS 20480"
) 
configs=(
"LeNet/conv1 0.58 $DEFAULT_MAX_NUMBER_OF_STEPS 800"
"LeNet/conv2 0.22 $DEFAULT_MAX_NUMBER_OF_STEPS 51200"
"LeNet/fc3   0.34 $DEFAULT_MAX_NUMBER_OF_STEPS 3211264"
"LeNet/fc4   0.36 $DEFAULT_MAX_NUMBER_OF_STEPS 10240"
) 

Total_size_of_variables=134281029

##!!!checkpoint_exclude_scopes is the last layer_name of the array configs by default.
checkpoint_exclude_scopes=`echo "${configs[0]}" | awk  '{print $1}'`
echo "checkpoint_exclude_scopes:"$checkpoint_exclude_scopes


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

#####################################################
#              Pruning Functions
#####################################################
function parse_configs()
{
    local line="${configs[$1]}"
    local para=`echo $line | awk -v col="$2" '{print $col}'`
    echo $para
}


#[...,bottom2,bottom1,currentLayer,top1,top2,...]
#sel_layers_bottom=sel_layers_top=0 means select currentLayer only.
function get_multilayer_scopes()
{
    local sel_layers_bottom=$1
    local sel_layers_top=$2
    local row=$3
    local col=$4
    local comma_scopes=""
    for((iter=$row-$sel_layers_bottom;iter<=$row+$sel_layers_top;iter++))
    do
	iter_scope=`parse_configs $iter $col`
	if [  $iter -ge 0  -a  -n "$iter_scope" ]
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
	exit -1
    fi
    _ckpt_=`cat $train_dir/checkpoint | grep -v all_model_checkpoint_paths | awk '{print $2}'`
    ckpt_=${_ckpt_#\"}
    ckpt=${ckpt_%\"}
    checkpoint_path=$train_dir/$ckpt
    echo $checkpoint_path
}

#####################################################
#               Eval Functions
#####################################################
function _eval_image_classifier()
{
    local train_dir=$1
    local tmp_checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
    echo "train_dir="$train_dir
    echo "tmp_checkpoint_path="$tmp_checkpoint_path
    python3 eval_image_classifier.py --alsologtostderr --checkpoint_path=${tmp_checkpoint_path} --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL \
	--model_name=$MODEL_NAME --eval_dir ${train_dir}/eval_event --labels_offset=$LABELS_OFFSET --max_num_batches=50 2>&1 | grep logging
}

function get_Recall_5()
{
    local result_str=`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Recall_5" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Recall_5() failed!"
	exit -1
    fi
    echo $result_mul_10000
}

# Usage: Accuracy=`get_Accuracy $train_dir`
function get_Accuracy()
{
    local result_str=`_eval_image_classifier $1`
    local result=`echo $result_str | awk -F "Accuracy" '{print $2}' | awk -F "[" '{print $2}' | awk -F "]" '{print $1}'`
    local result_mul_10000=`echo "scale=0;$result*10000/1"|bc`
    if [ -z "$result_mul_10000" ]
    then
	echo "Error:envoke get_Accuracy() failed!"
	exit -1
    fi
    echo $result_mul_10000
}



#####################################################


function modify_string()
{
    local multiLayers=$1      #multiLayers=vgg_16/conv1/conv1_1,vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_1
    local multiLayers_rate=$2 #multiLayers_rate=0.58,0.22,0.34
    local modifyLayer=$3      #modifyLayer=vgg_16/conv1/conv1_2
    local modifyRate=$4       #modifyRate=0.5
                        #output : 0.58,0.5,0.34
    local NF=`echo "$multiLayers" | awk -F "," '{print NF}'`
    local _rates=""
    for((i=1;i<=$NF;i+=1))
    do
	curLayer=`echo "$multiLayers" | awk -v col=$i -F "," '{print $col}'`
	curRate=`echo "$multiLayers_rate" | awk -v col=$i -F "," '{print $col}'`
	if [ $curLayer = $modifyLayer ]
	then
	    _rates=$_rates,$modifyRate
	else
	    _rates=$_rates,$curRate
	fi
    done
    _rates=${_rates#,}
    echo "$_rates"
}

function parse_args()
{
    arg=`echo -n "$1" | awk -F "${2}=" '{print $NF}' | awk -F " " '{print $1}'`
    echo "$arg"
}


function print_info()
{
    echo "######################################################"
    echo "pruning_and_retrain_step:" $*
    echo "CHECKPOINT_PATH:" $checkpoint_path "TRAIN_DIR:" $train_dir
    echo "######################################################"    
    #echo "SECONDS="$SECONDS
}

function remove_redundant_cpkt()
{
    local train_dir=$1
    local tmp_checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
    ls $(dirname $tmp_checkpoint_path)/model.ckpt* | grep -v ${tmp_checkpoint_path}. | xargs rm -rvf
    echo "remaining:"
    ls $(dirname $tmp_checkpoint_path)/model.ckpt* -l
}

function pruning_and_retrain_step_eval()
{
    #global DATASET_DIR MODEL_NAME
    local max_number_of_steps=`parse_args "$*" "max_number_of_steps"`
    local train_dir=`parse_args "$*" "train_dir"`

    if [ -z $max_number_of_steps ]
    then
	max_number_of_steps=20
    fi
    if [ -f $train_dir/checkpoint ]
    then
	cur_step=`cat $train_dir/checkpoint | grep -v "all_model_checkpoint_paths" | awk -F "-|\"" '{print $3}'`
	if [  $cur_step -ge $max_number_of_steps ]
	then
	    return -1
	fi
    fi

    local _pre_pass_Accuracy=0
    local cnt=0
    for((consum_number_of_steps=10;consum_number_of_steps<=$max_number_of_steps;consum_number_of_steps+=$EVAL_INTERVAL))
    do
	echo "eval command:" $@
	echo "max_number_of_steps:" $consum_number_of_steps
	pruning_gradient_update_ratio=0
	if [ $En_AUTO_RATE_PRUNING_EARLY_SKIP = "Enable" ]
	then
	    if [ $cnt -eq 8 -o $cnt -eq 12 -o $cnt -eq 16 ] #
	    then
		pruning_gradient_update_ratio=10 #close
		echo "Current consum_number_of_steps =" $consum_number_of_steps
		echo "pruning_gradient_update_ratio  =" $pruning_gradient_update_ratio
	    fi
	fi
	if [ $consum_number_of_steps -gt  $max_number_of_steps ]
	then
	    consum_number_of_steps=$max_number_of_steps
	fi

	python3 train_image_classifier.py --noclone_on_cpu --optimizer sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	    --save_summaries_secs=$SAVE_SUMMARIES_SECS $@ \
	    --max_number_of_steps=$consum_number_of_steps --pruning_gradient_update_ratio=$pruning_gradient_update_ratio

	#remove_redundant_cpkt $train_dir	

	g_Accuracy=`get_Accuracy $train_dir`
	g_Recall_5=`get_Recall_5 $train_dir`
	echo "g_Accuracy="$g_Accuracy
	echo "g_Recall_5="$g_Recall_5
	if [ -z "$g_Accuracy" ]
	then
	    echo "Error!"
	    exit -1
	fi
	if [ $consum_number_of_steps -eq 10 ]
	then
	    consum_number_of_steps=0
	else
	    echo "g_Accuracy_thr="$g_Accuracy_thr
	    echo "g_Accuracy="$g_Accuracy
	    if [ $En_AUTO_RATE_PRUNING_EARLY_SKIP = "Enable" ]
	    then
		if [ $g_Accuracy_thr -le $g_Accuracy -a $pruning_gradient_update_ratio -eq 0 ]
		then
		    if [ $_pre_pass_Accuracy -ge $g_Accuracy ]
		    then
			return 0
		    fi
		    _pre_pass_Accuracy=$g_Accuracy
		fi	
	    fi    
	fi
	let "cnt+=1"
    done
}


function auto_rate_pruning()
{

    pruning_scopes=`parse_args "$*" "pruning_scopes"`
    pruning_rates=`parse_args "$*" "pruning_rates"`
    local pruning_step=10
    local allow_pruning_loss=20 #0.2%*100
    local begin_pruning_rate=80
    let "g_Accuracy_thr=g_preAccuracy-allow_pruning_loss"

    g_prePass_rate=$begin_pruning_rate
    g_prePass_checkpoint_path=$checkpoint_path

    local cnt_step=5
    local local_cnt=0
    local count=0
    local layer_train_dir=$train_dir
    for((rate100=$begin_pruning_rate;rate100>0;rate100-=$pruning_step))
    do
	#local rate="0.$rate100"
	local rate=$(echo "scale=2; $rate100 / 100"  | bc)
	scopes_rate=`modify_string $pruning_scopes $pruning_rates ${pruning_scopes%%,*} $rate`


	echo -e "\n\n"
	rate_train_dir=${layer_train_dir}_${count}_$rate100
	#rm ${layer_train_dir}_*_$rate100 -rvf #delate failed tries
	pruning_and_retrain_step_eval $@ --pruning_rates=$scopes_rate \
            --checkpoint_path=${checkpoint_path}  --train_dir=${rate_train_dir}

	echo -e "Round "$count "Result:"
	echo "g_Accuracy="$g_Accuracy
	echo "g_preAccuracy="$g_preAccuracy
	echo "g_Accuracy_thr="$g_Accuracy_thr
	echo "local_cnt="$local_cnt
	echo "rate="$rate

	if [ "$g_Accuracy" -lt $g_Accuracy_thr ]
	then
	    let "rate100=rate100+pruning_step" #this is right
	    let "pruning_step=pruning_step/2"
	    let "local_cnt+=1"
	    echo "Draw Back $local_cnt Times."
	else
	    checkpoint_path=`next_CHECKPOINT_PATH $rate_train_dir`
	    g_prePass_checkpoint_path=$checkpoint_path
	    g_prePass_rate=$rate100
	    echo -e "Pass."
            if [ $rate100 -le $pruning_step ]
            then
                let "pruning_step=pruning_step/2"
            fi
	fi
	echo -e "pruning_step="$pruning_step " \n\n"

	if [ $local_cnt -ge $cnt_step -o $pruning_step -eq 0 ]
	then
	    let "rate100=rate100+pruning_step"
	    break
	fi
	let "count+=1"
    done
    pruned_dir=`dirname $g_prePass_checkpoint_path`    
    mv $pruned_dir ${train_dir} -v
    mkdir -p ${pruned_dir}_final

    echo -e "The final pruning rate of layer $(basename $train_dir) is:"$g_prePass_rate  #output rate
    g_preAccuracy=$g_Accuracy
    echo -e "The final Accuracy of layer $(basename $train_dir) is:"$g_Accuracy "\n\n" #for next layer
}



#####################################################
#               Pruning Process Flow
#####################################################
#(A)Retrain for Convergence / Train from Scratch
g_preAccuracy=9999 # init
En_AUTO_RATE_PRUNING_EARLY_SKIP="Disable"
train_dir=${TRAIN_DIR_PREFIX}_${MODEL_NAME}/Retrain_from_Scratch
print_info "A"
#pruning_and_retrain_step_eval --train_dir=${train_dir} \
#    --learning_rate=0.01  --weight_decay=0.0005 --batch_size=64 --max_number_of_steps=16000 \
    #--learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64 --max_number_of_steps=50 \

    ##--checkpoint_path=${checkpoint_path}
    #####--checkpoint_exclude_scopes=$checkpoint_exclude_scopes --trainable_scopes=$checkpoint_exclude_scopes \
#checkpoint_path=`next_CHECKPOINT_PATH $train_dir`

#checkpoint_path=./train_dir_AB_mnist_rmsprop/lenet/Retrain_from_Scratch/model.ckpt-16000
#g_Accuracy=9728
#g_Accuracy=9786
#train_dir=`dirname $checkpoint_path`
#echo "init model path:$train_dir"
#g_Accuracy=`get_Accuracy $train_dir`
#Calculate and Print Eval Info
g_preAccuracy=$g_Accuracy
echo "checkpoint_path :" $checkpoint_path
echo "g_preAccuracy =" $g_preAccuracy
#echo "Recall_5 =" $Recall_5

#(B)Pruning without Retrain
if [ "$En_AUTO_RATE_PRUNING_WITHOUT_RETRAIN" = "EnableX" ]
then

    En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable"
    g_train_dir=${TRAIN_DIR_PREFIX}_${MODEL_NAME}/Pruning_and_Retrain_Layer_by_Layer
    row=0
    total_row_num=${#configs[@]}
    let "total_row_num-=1"
    #for row in `seq $total_row_num -1 0`
    for row in `seq 0 1 $total_row_num`
    #for line in "${configs[@]}"
    do
	line=${configs[$row]}
	layer_name=`echo $line | awk '{print $1}'`
	pruning_rate=`echo $line | awk '{print $2}'`
	max_number_of_steps=`echo $line | awk '{print $3}'`
	
        #Pruning without Retrain
	train_dir=${g_train_dir}
	print_info "B-$row" ; echo "Current config line --- configs[$row]:" $line

	train_dir=${g_train_dir}/$layer_name

	trainable_scopes=$layer_name
        pruning_scopes_pyramid="$pruning_scopes_pyramid,$layer_name"
	pruning_rates_pyramid="$pruning_rates_pyramid,$pruning_rate"

	trainable_scopes=`get_multilayer_scopes 0 0 $row 1`
	pruning_scopes=`get_multilayer_scopes 0 0 $row 1`
	pruning_rates=`get_multilayer_scopes 0 0 $row 2`
	
	auto_rate_pruning --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} --max_number_of_steps=$max_number_of_steps \
	    --trainable_scopes=$trainable_scopes --pruning_scopes=$pruning_scopes --pruning_rates=$pruning_rates --pruning_strategy=AUTO \
	    --learning_rate=0.001  --weight_decay=0.0005 --batch_size=64

	checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
	echo "checkpoint_path is:" $checkpoint_path
	if [ $row -eq -1 ]
	then
	    exit 0
	fi
    done
    exit 0
    checkpoint_path=`next_CHECKPOINT_PATH $train_dir`
fi

#(C)Pruning and Retrain

####################################################################################
##########             pruning_and_retrain_multilayers_iter               ##########
####################################################################################
trainable_scopes_pyramid=""
pruning_rates_pyramid=""
total_row_num=${#configs[@]}
let "total_row_num-=1"
for row in `seq $total_row_num -1 0`
#for row in `seq 0 1 $total_row_num`
do
    line=${configs[$row]}
    layer_name=`echo $line | awk '{print $1}'`
    pruning_rate=`echo $line | awk '{print $2}'`
    #max_number_of_steps=`echo $line | awk '{print $3}'`
    trainable_scopes_pyramid="$trainable_scopes_pyramid,$layer_name"
    pruning_rates_pyramid="$pruning_rates_pyramid,$pruning_rate"
done
trainable_scopes_pyramid=${trainable_scopes_pyramid#,}
pruning_rates_pyramid=${pruning_rates_pyramid#,}

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


function pruning_and_retrain_step_eval_multiLayer()
{
    #global DATASET_DIR MODEL_NAME
    local max_number_of_steps=`parse_args "$*" "max_number_of_steps"`
    local train_dir=`parse_args "$*" "train_dir"`
    local pruning_rates=`parse_args "$*" "pruning_rates"`
    if [ -z $max_number_of_steps ]
    then
	max_number_of_steps=$EVAL_INTERVAL
    fi
    if [ $max_number_of_steps -lt $EVAL_INTERVAL ]
    then
	max_number_of_steps=$EVAL_INTERVAL
    fi
    
    local starting_step=10
    if [ -f $train_dir/checkpoint ]
    then
	cur_step=`cat $train_dir/checkpoint | grep -v "all_model_checkpoint_paths" | awk -F "-|\"" '{print $3}'`
	let "starting_step=starting_step+cur_step"
	echo "cur_step="$cur_step
	echo "starting_step="$starting_step
	if [  $cur_step -ge $max_number_of_steps ]
	then
	    echo "Error:envoke pruning_and_retrain_step_eval_multiLayer() failed!"
	    echo "Error: cur_step=$cur_step ge max_number_of_steps=$max_number_of_steps "
	    return -1
	fi
    fi

    local _pre_pass_Accuracy=0
    local cnt=0
    for((consum_number_of_steps=$starting_step;consum_number_of_steps<=$max_number_of_steps;consum_number_of_steps+=$EVAL_INTERVAL))
    do
	echo "eval command:" $@
	echo "max_number_of_steps:" $consum_number_of_steps
	pruning_gradient_update_ratio=0
	if [ $En_AUTO_RATE_PRUNING_EARLY_SKIP = "Enable" ]
	then
	    if [ $cnt -eq 8 -o $cnt -eq 12 -o $cnt -eq 16 ] #
	    then
		pruning_gradient_update_ratio=0 #close
		echo "Current consum_number_of_steps =" $consum_number_of_steps
		echo "pruning_gradient_update_ratio  =" $pruning_gradient_update_ratio
	    fi
	fi
	if [ $consum_number_of_steps -gt  $max_number_of_steps ]
	then
	    consum_number_of_steps=$max_number_of_steps
	fi
	python3 train_image_classifier.py --noclone_on_cpu --optimizer sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	    --save_summaries_secs=$SAVE_SUMMARIES_SECS $@ \
	    --max_number_of_steps=$consum_number_of_steps --pruning_gradient_update_ratio=$pruning_gradient_update_ratio

	#remove_redundant_cpkt $train_dir	

	g_Accuracy=`get_Accuracy $train_dir`
	g_Recall_5=`get_Recall_5 $train_dir`
	echo "train_dir Pass for eval:=$train_dir"
	echo "g_preAccuracy="$g_preAccuracy
	echo "g_Accuracy_thr="$g_Accuracy_thr
	echo "g_Accuracy="$g_Accuracy
	echo "g_Recall_5="$g_Recall_5
	echo "pruning_rates="$pruning_rates    
	if [ $consum_number_of_steps -eq 10 ]
	then
	    consum_number_of_steps=0
	else
	    if [ $En_AUTO_RATE_PRUNING_EARLY_SKIP = "Enable" ]
	    then
		if [ $g_Accuracy_thr -le $g_Accuracy -a $consum_number_of_steps -ne 0 -a $pruning_gradient_update_ratio -eq 0 ]
		then
		    echo "Pass!"
		    return 0
		    if [ $_pre_pass_Accuracy -ge $g_Accuracy ]
		    then
			return 0
		    fi
		    _pre_pass_Accuracy=$g_Accuracy
		fi	
		echo "Continue!"
	    fi    
	fi
	let "cnt+=1"
    done
    return 1
}

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
	cur_rates="0.8"
	cur_steps="0.10"
	for((i=1;i<$all_trainable_scopes_num;i+=1))
	do
	    cur_rates="$cur_rates,0.80"    
	    cur_steps="$cur_steps,0.10"    
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

  
    local pruning_layers_num=`echo $pruning_layers_index | awk -F "," '{print NF}'`
    let "g_Accuracy_thr=g_preAccuracy-allow_pruning_loss"

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
    
    count=0
    local all_trainable_scopes_num=`echo $all_trainable_scopes | awk -F "," '{print NF}'`
    echo "multilayers_iter:all_trainable_scopes_num=$all_trainable_scopes_num" ##
    echo "multilayers_iter:all_trainable_scopes=$all_trainable_scopes" ##
    for((col=0;col<all_trainable_scopes_num;col+=1))
    do
	echo -e "\n\nmultilayers_iter:col=$col"
	echo "train_dir="$train_dir
	echo "checkpoint_path="$checkpoint_path
	echo "all_trainable_scopes="$all_trainable_scopes
	
	echo "pruning_layers_index="$pruning_layers_index

	is_substr "$pruning_layers_index" $col
	if [ $? -eq 1 ]
	then
	    echo "this layer is not a prunable layer. skip!"
	    echo "col="$col
	    echo "pruning_layers_index="$pruning_layers_index
	    continue
	fi
	pruning_rates=`read_from_file $ckhp_iter_Rates_txt`
	echo "pruning_rates(passed)=$pruning_rates"
	pruning_steps=`read_from_file $ckhp_iter_Steps_txt`
	pruning_rate=`get_str $pruning_rates $col`
	pruning_step=`get_str $pruning_steps $col`
	pruning_rate=`echo "scale=2;$pruning_rate-$pruning_step/1.0"|bc`
	echo "pruning_rate=$pruning_rate"
	pruning_rates=`modify_str $pruning_rates $col  $pruning_rate` 
	echo "pruning_steps=$pruning_steps"
	smaller=`awk -v numa=0.01 -v numb=$pruning_step 'BEGIN{print(numa>numb)?"1":"0"}'`
	if [ $smaller -eq 1 ]
	then
	    let "count+=1"
	    continue
	fi

	local max_number_of_steps=`get_cur_number_of_steps ${train_dir}_pass`
	echo "get_cur_number_of_steps:max_number_of_steps="$max_number_of_steps
	let "max_number_of_steps+=pruning_singlelayer_retrain_step"
	
	echo "pruning_layers_index_last="${pruning_layers_index##*,}
	if [ ${pruning_layers_index##*,} -eq $col ]
	then
	    let "max_number_of_steps+=pruning_multilayers_retrain_step"
	fi

	pruning_and_retrain_step_eval_multiLayer --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} \
            --trainable_scopes=$all_trainable_scopes --pruning_scopes=$all_trainable_scopes \
            --pruning_rates=$pruning_rates --max_number_of_steps=$max_number_of_steps --pruning_strategy=ABS \
            --learning_rate=0.001  --weight_decay=0.0005 --batch_size=64  #2>&1 >> /dev/null
	if [ $? -ne 0 ]
	then
	    #is_curTry_Pass="False"
	    pruning_rate=`echo "scale=2;$pruning_rate+$pruning_step/1.0"|bc`
	    pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
	    rm ${train_dir} -rf
	    checkpoint_dir=`dirname $checkpoint_path`
	    cp $checkpoint_dir ${train_dir} -rf
	    pruning_steps=`modify_str $pruning_steps $col  $pruning_step` 
	    write_to_file $ckhp_iter_Steps_txt "$pruning_steps"
	    cp $ckhp_iter_Steps_txt $checkpoint_dir -rf
	else
	    #is_curTry_Pass="True"
	    write_to_file $ckhp_iter_Rates_txt "$pruning_rates"
	    rm ${train_dir}_pass -rf
	    cp ${train_dir} ${train_dir}_pass -rf
	    checkpoint_path=`next_CHECKPOINT_PATH ${train_dir}_pass`

	    smaller=`awk -v numa=$pruning_rate -v numb=$pruning_step 'BEGIN{print(numa<=numb)?"1":"0"}'`
            if [ $smaller -eq 1 ]
            then
		pruning_step=`echo "scale=2;$pruning_step/2.0"|bc`
            fi
	fi    
	
	
	if [ $col -eq $all_trainable_scopes_num -a 1 -eq 0 ]
	then
            col=-1
	fi
    done 
    if [ $count -lt $all_trainable_scopes_num ] #check if all pruning_steps all zero.
    then 
	if [ ! -d ${train_dir}_pass ]
	then
            let "pre_iter=iter-1"
            echo "pre_iter:"$pre_iter
            cp ${train_Dir}/iter${pre_iter}_pass ${train_dir}_pass -rv
            cp $ckhp_iter_Rates_txt ${train_dir}_pass -rf
            cp $ckhp_iter_Steps_txt ${train_dir}_pass -rf
	fi
	rm ${train_dir} -rf
	write_to_file $ckhp_iter_PC_txt $iter
    fi
}


####################################################################################
##########               pruning_and_retrain_multilayers                  ##########
####################################################################################

#multiLayers iters rate_decay


function pruning_and_retrain_multilayers()
{
    local all_trainable_scopes=$1
    local max_iters=$2
    local allow_pruning_loss=$3 #0.2%*100
    local train_Dir=$4
    local checkpoint_Path=$5
    local pruning_layers_index=$6

    pruning_singlelayer_retrain_step=200
    pruning_multilayers_retrain_step=1000
    

    ckhp_iter_PC_txt=${train_Dir}/ckhp_iter_PC.txt
    local passed_iter=`get_cur_iter $ckhp_iter_PC_txt `
    echo "passed_iter=$passed_iter"
    for((iter=$passed_iter;iter<=$max_iters;iter+=1))
    do
	pruning_and_retrain_multilayers_iter $all_trainable_scopes -1 $allow_pruning_loss $train_Dir $checkpoint_Path $pruning_layers_index $pruning_singlelayer_retrain_step $pruning_multilayers_retrain_step
    done
}


checkpoint_path=./mnist_Train_from_Scratch_lenet/Retrain_from_Scratch/model.ckpt-15500
dir=`dirname $checkpoint_path`
echo $dir
#g_Accuracy=`get_Accuracy $dir`
g_Accuracy=9850
g_preAccuracy=$g_Accuracy
echo "g_Accuracy=$g_Accuracy"
echo "g_preAccuracy=$g_preAccuracy"
TRAIN_DIR_PREFIX=./train_dir_multiLayers
train_Dir=${TRAIN_DIR_PREFIX}_${MODEL_NAME}/Retrain_Prunned_Network

echo "#####################################################################"

#trainable_scopes_pyramid
all_trainable_scopes=LeNet/fc4,LeNet/fc3,LeNet/conv2,LeNet/conv1
pruning_layers_index="0,1,3"
max_iters=20
allow_pruning_loss=50
#train_Dir
#checkpoint_path
echo "$all_trainable_scopes $max_iters $allow_pruning_loss $train_Dir $checkpoint_path"
#enovke
pruning_and_retrain_multilayers $all_trainable_scopes $max_iters $allow_pruning_loss $train_Dir $checkpoint_path $pruning_layers_index

exit 0
