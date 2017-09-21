#!/bin/bash
trap "kill 0" INT

source ./config_mnist.sh
source ./common_function.sh
source ./eval_function.sh
source ./train_function.sh

#train_dir check_dir fdata_dir tdata_dir max_steps 
#pruning_scopes pruning_steps train_scopes 
function prune_net()
{
    echo "prune_net"
    return $?
}

#train_dir evalt_loss_anc evalt_loss_cur evalt_loss_thr
function evalt_net() 
{
    echo "evalt_net"
    return $?
}

#max_steps evalt_interval
function prune_and_evalt_step()
{
    local max_steps=$1
    local evalt_interval=${2:-50}
    local is_early_skip=${2:-"early_skip"}
    for((step=0;step<$max_steps;step+=$evalt_interval))
    do
	echo "prune_and_evalt_step::step="$step "evalt_interval="$evalt_interval
	prune_net
	evalt_net
    done
}

#pruning_scopes pruning_rate_drop_step pruning_singlelayer_retrain_step pruning_multilayers_retrain_step
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

