#!/bin/bash
trap "kill 0" INT

#####################################################
#                 Global Config
#####################################################
DATASET_DIR=/tmp/mnist #/mllib/ImageNet/ILSVRC2012_tensorflow
DATASET_NAME=mnist     #imagenet
TRAIN_DIR_PREFIX=./train_dir_AB_mnist_weight_decay
EVAL_INTERVAL=250
SAVE_SUMMARIES_SECS=250
DEFAULT_MAX_NUMBER_OF_STEPS=5000
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
    python eval_image_classifier.py --alsologtostderr --checkpoint_path=${tmp_checkpoint_path} --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=$DATASET_SPLIT_NAME_FOR_VAL \
	--model_name=$MODEL_NAME --eval_dir ${train_dir}/eval_event --labels_offset=$LABELS_OFFSET --max_num_batches=50 2>&1 | grep logging
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

function pruning_and_retrain_step_eval()
{
    #global DATASET_DIR MODEL_NAME
    #max_number_of_steps=`echo -n "$*" | awk -F "max_number_of_steps=" '{print $NF}' | awk -F " " '{print $1}'`
    #train_dir=`echo -n "$*" | awk -F "train_dir=" '{print $NF}' | awk -F " " '{print $1}'`
    max_number_of_steps=`parse_args "$*" "max_number_of_steps"`
    train_dir=`parse_args "$*" "train_dir"`

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
    for((consum_number_of_steps=1000;consum_number_of_steps<=$max_number_of_steps;consum_number_of_steps+=$EVAL_INTERVAL))
    do
	echo "eval command:" $@
	echo "max_number_of_steps:" $consum_number_of_steps
	pruning_gradient_update_ratio=0
	if [ $cnt -eq 8 -o $cnt -eq 12 -o $cnt -eq 16 ]
	then
	    pruning_gradient_update_ratio=10 #close
	fi
	if [ $consum_number_of_steps -gt  $max_number_of_steps ]
	then
	    consum_number_of_steps=$max_number_of_steps
	fi

	python train_image_classifier.py --noclone_on_cpu --optimizer sgd --labels_offset=$LABELS_OFFSET --dataset_dir=${DATASET_DIR} --dataset_name=$DATASET_NAME --dataset_split_name=train --model_name=$MODEL_NAME \
	    --save_summaries_secs=$SAVE_SUMMARIES_SECS $@ \
	    --max_number_of_steps=$consum_number_of_steps --pruning_gradient_update_ratio=$pruning_gradient_update_ratio
	
	local result_str=`eval_image_classifier $train_dir`
	g_Accuracy=`get_Accuracy $result_str`
	g_Recall_5=`get_Recall_5 $result_str`
	echo "g_Accuracy="$g_Accuracy
	echo "g_Recall_5="$g_Recall_5
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

    local begin_pruning_rate=90
    g_prePass_rate=$begin_pruning_rate
    local count_step=4
    let "g_Accuracy_thr=g_preAccuracy-allow_pruning_loss"
    local local_cnt=0
    local count=0
    local layer_train_dir=$train_dir
    for((rate100=$begin_pruning_rate;rate100>0;rate100-=$pruning_step))
    do
	#local rate="0.$rate100"
	local rate=$(echo "scale=2; $rate100 / 100"  | bc)
	scopes_rate=`modify_string $pruning_scopes $pruning_rates ${pruning_scopes%%,*} $rate`
	echo "scopes_rate="$scopes_rate

	echo -e "\n\n"
	rate_train_dir=${layer_train_dir}_$rate100
	rm $rate_train_dir -rf
	pruning_and_retrain_step_eval $@ --pruning_rates=$scopes_rate \
            --checkpoint_path=${checkpoint_path}  --train_dir=${rate_train_dir}

	echo -e "Round "$count "Result:"
	echo "g_Accuracy="$g_Accuracy
	echo "g_preAccuracy="$g_preAccuracy
	echo "g_Accuracy_thr="$g_Accuracy_thr
	echo "local_cnt="$local_cnt
	echo "rate="$rate

	if [ -z "$g_Accuracy" ]
	then
	    echo "Error!"
	    exit -1
	fi
	if [ "$g_Accuracy" -lt $g_Accuracy_thr ]
	then
	    let "rate100=rate100+pruning_step" #this is right
	    let "pruning_step=pruning_step/2"
	    let "local_cnt+=1"
	    echo "Draw Back $local_cnt Times."
	else
	    checkpoint_path=`next_CHECKPOINT_PATH $rate_train_dir`
	    g_prePass_rate=$rate100
	    echo "Pass."
            if [ $rate100 -le $pruning_step ]
            then
                let "pruning_step=pruning_step/2"
            fi
	fi
	echo "pruning_step="$pruning_step

	if [ $local_cnt -ge $count_step -o $pruning_step -eq 0 ]
	then
	    let "rate100=rate100+pruning_step"
	    break
	fi
	let "count+=1"
    done
    echo -e "Finally the compress rate is:\n g_prePass_rate="$g_prePass_rate  #output rate
    echo "checkpoint_path="$checkpoint_path
    g_preAccuracy=$g_Accuracy
    echo -e "g_preAccuracy="$g_Accuracy "\n\n" #for next layer
}

#####################################################
#               Pruning Process Flow
#####################################################
#(A)Retrain for Convergence / Train from Scratch
g_preAccuracy=9999 # init
En_AUTO_RATE_PRUNING_EARLY_SKIP="Disable"
train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Retrain_from_Scratch
print_info "A"
###pruning_and_retrain_step_eval --train_dir=${train_dir} \
   ### --learning_rate=0.01  --weight_decay=0.00005 --batch_size=64 --max_number_of_steps=16000 \
    #--learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64 --max_number_of_steps=50 \

    ##--checkpoint_path=${checkpoint_path}
    #####--checkpoint_exclude_scopes=$checkpoint_exclude_scopes --trainable_scopes=$checkpoint_exclude_scopes \
checkpoint_path=`next_CHECKPOINT_PATH $train_dir`

checkpoint_path=./train_dir_AB_mnist/lenet/Retrain_from_Scratch/model.ckpt-16000
g_Accuracy=9728
#Calculate and Print Eval Info
g_preAccuracy=$g_Accuracy
echo "checkpoint_path :" $checkpoint_path
echo "g_preAccuracy =" $g_preAccuracy
#echo "Recall_5 =" $Recall_5

#(B)Pruning without Retrain
if [ "$En_AUTO_RATE_PRUNING_WITHOUT_RETRAIN" = "Enable" ]
then

    En_AUTO_RATE_PRUNING_EARLY_SKIP="Enable"
    g_train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Pruning_without_Retrain
    row=0
    total_row_num=${#configs[@]}
    let "total_row_num-=1"
    for row in `seq $total_row_num -1 0`
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
exit 0
#(C)Pruning and Retrain
trainable_scopes_pyramid=""
pruning_rates_pyramid=""
row=0
total_row_num=${#configs[@]}
let "total_row_num-=1"
for row in `seq $total_row_num -1 0`
#for row in `seq 0 1 $total_row_num`
do
    line=${configs[$row]}
    layer_name=`echo $line | awk '{print $1}'`
    pruning_rate=`echo $line | awk '{print $2}'`
    max_number_of_steps=`echo $line | awk '{print $3}'`
    if [ -z $trainable_scopes_pyramid ]
    then
	trainable_scopes_pyramid="$layer_name"
	pruning_rates_pyramid="$pruning_rate"
    else
	trainable_scopes_pyramid="$trainable_scopes_pyramid,$layer_name"
	pruning_rates_pyramid="$pruning_rates_pyramid,$pruning_rate"
    fi

    #Pruning and Retrain
    train_dir=$TRAIN_DIR_PREFIX/$layer_name
    if [ $row -ge $REPRUNING_FROM_LAYER_TH -a "$En_REPRUNING_FROM_SPECIFIC_LAYER" = "Enable" ]
    then
	echo -e "\n\nRepruning $layer_name"
	rm $train_dir -rf #!!!!
    fi
    print_info "C-$row" ; echo "Current config line --- configs[$row]:" $line

    trainable_scopes=`get_multilayer_scopes 0 0 $row 1`
    pruning_rates=`get_multilayer_scopes 0 0 $row 2`

    pruning_and_retrain_step_eval --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} --max_number_of_steps=$max_number_of_steps \
	--trainable_scopes=$trainable_scopes --pruning_scopes=$trainable_scopes --pruning_rates=$pruning_rates \
	--learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64 
    
    checkpoint_path=`next_CHECKPOINT_PATH $train_dir`

done
#trainable_scopes_pyramid=${trainable_scopes_pyramid#,}
#pruning_rates_pyramid=${pruning_rates_pyramid#,}

#[D]Exit for Manual Modification


#(E)Retrain the Prunned Network 
checkpoint_path=./train_dir_ACE/vgg_16/fc8/model.ckpt-5000
TRAIN_DIR_PREFIX=./train_dir_ACE_retrain_to_10000
train_dir=$TRAIN_DIR_PREFIX/$MODEL_NAME/Retrain_Prunned_Network
print_info "E"
pruning_and_retrain_step_eval --checkpoint_path=${checkpoint_path}  --train_dir=${train_dir} \
    --trainable_scopes=$trainable_scopes_pyramid --pruning_scopes=$trainable_scopes_pyramid --pruning_rates=$pruning_rates_pyramid --max_number_of_steps=10000 \
    --learning_rate=0.00001  --weight_decay=0.00005 --batch_size=64 
