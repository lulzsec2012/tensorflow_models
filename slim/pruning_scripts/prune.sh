#!/bin/bash
trap "kill 0" INT

source ./train_evalt_function.sh

echo $CUDA_VISIBLE_DEVICES
exit 0
#train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_rates train_scopes 
function prune_net()
{
    echo "prune_net..."
    for((i=0;i<3;i+=1))
    do
	local nvidia_avaiable="True"
	if [ $nvidia_avaiable = "True" ]
	then
	    train_program
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
	    evalt_program
	    if [ $? -eq 0 ]
	    then
		return 0
	    fi
	fi
	sleep 60s
    done
    return 1
}


train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_rates train_scopes 
prune_net
exit 0
#Real eval command: python3 train_image_classifier.py --noclone_on_cpu --optimizer=sgd --labels_offset=1 --dataset_dir=/mllib/ImageNet/ILSVRC2012_tensorflow             --dataset_name=imagenet --dataset_split_name=train --model_name=vgg_16 	    --save_summaries_secs=1000 --checkpoint_path=./train_dir_multiLayers_imagenet_from_50000_learning_rate0.0001_reconfigGAccuracy_allowBiasBp_worker_replicas2_test_num_clones1_vgg_16/Retrain_Prunned_Network/iter0_pass/model.ckpt-50000 --train_dir=./train_dir_multiLayers_imagenet_from_50000_learning_rate0.0001_reconfigGAccuracy_allowBiasBp_worker_replicas2_test_num_clones1_vgg_16/Retrain_Prunned_Network/iter1 --trainable_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5/conv5_3,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_1,vgg_16/conv4/conv4_3,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_1,vgg_16/conv3/conv3_3,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_1,vgg_16/conv2/conv2_2,vgg_16/conv2/conv2_1,vgg_16/conv1/conv1_2,vgg_16/conv1/conv1_1 --pruning_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5/conv5_3,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_1,vgg_16/conv4/conv4_3,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_1,vgg_16/conv3/conv3_3,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_1,vgg_16/conv2/conv2_2,vgg_16/conv2/conv2_1,vgg_16/conv1/conv1_2,vgg_16/conv1/conv1_1 --pruning_rates=.96,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00 --max_number_of_steps=100010 --pruning_strategy=ABS --log_every_n_steps=10 --save_interval_secs=600 --momentum=0.9 --end_learning_rate=0.00001 --learning_rate=0.0001 --weight_decay=0.0005 --batch_size=64 	    --max_number_of_steps=2000 --pruning_gradient_update_ratio=0 --num_clones=1

#train_dir check_dir  max_steps 
#max_steps evalt_interval  g_early_skip
function prune_and_evalt_step()
{
    for((step=0;step<$max_steps;step+=$evalt_interval))
    do
	echo "prune_and_evalt_step::step="$step "evalt_interval="$evalt_interval
	prune_net
	if [ $g_early_skip = "True" ]
	then
	    max_steps=
	    evalt_net
	    if [ $g_evalt_loss_pass = "True" ]
	    then
		break
	    fi
	fi
    done
    if [ $g_early_skip = "False" ]
    then
	evalt_net
    fi
    if [  $g_evalt_loss_pass = "True" ]
    then
	mv $train_dir ${train_dir}_pass
	return 0
    else
	rm $train_dir -rf
	return 1
    fi
}

#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step
#pruning_layers_index="0 1 2 3"
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
function prune_and_evalt_scope()
{
    for scope in ``
    do
	prune_and_evalt_step
    done
}

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

