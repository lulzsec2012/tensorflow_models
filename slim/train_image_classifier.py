# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training.optimizer import _get_processor as get  
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


slim = tf.contrib.slim

###add by lzlu
import numpy as np
tf.app.flags.DEFINE_string(
    'pruning_scopes', None,
    'Comma-separated list of pruning scope.'
    'By default, None would prunned.')
tf.app.flags.DEFINE_string(
    'pruning_rates', None,
    'Comma-separated list of pruning rates used to prun each trainable scope.'
    'By default, The default pruning rate is 1.0.')
tf.app.flags.DEFINE_string(
    'pruning_strategy', 'AUTO', 'The name of the strategy used to prun.')
tf.app.flags.DEFINE_integer(
    'pruning_gradient_update_ratio', 0,
    'The update ratio of the pruned gradients. 0 for pruning,!=1 for DSD.')
tf.app.flags.DEFINE_boolean(
    'forbid_bias_bp', False,
    'forbid bias bp when training.')
###
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool( 
    'sync_replicas', False, 
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)
  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train
###add by lzlu
def apply_pruning_to_grad_bak(clones_gradients,pruningMask):
  """
  clones_gradients: [(<tf.Tensor 'gradients/AddN:0' shape=(1, 1, 4096, 5) dtype=float32>, <tf.Variable 'vgg_16/fc8/weights:0' shape=(1, 1, 4096, 5) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/fc8/BiasAdd_grad/tuple/control_dependency_1:0' shape=(5,) dtype=float32>, <tf.Variable 'vgg_16/fc8/biases:0' shape=(5,) dtype=float32_ref>), (<tf.Tensor 'gradients/AddN_1:0' shape=(1, 1, 4096, 4096) dtype=float32>, <tf.Variable 'vgg_16/fc7/weights:0' shape=(1, 1, 4096, 4096) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/fc7/BiasAdd_grad/tuple/control_dependency_1:0' shape=(4096,) dtype=float32>, <tf.Variable 'vgg_16/fc7/biases:0' shape=(4096,) dtype=float32_ref>), (<tf.Tensor 'gradients/AddN_2:0' shape=(3, 3, 3, 64) dtype=float32>, <tf.Variable 'vgg_16/conv1/conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/conv1/conv1_1/BiasAdd_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'vgg_16/conv1/conv1_1/biases:0' shape=(64,) dtype=float32_ref>)]
  """
  ##print("######################apply_pruning_to_grad:###############################")
  if pruningMask is None:
    return clones_gradients
  for mask_name,mask in pruningMask:
    count = 0
    assign_ops=[]
    for grad,var in clones_gradients:
      if var.name == mask_name:
        ###print("grad.name:",grad.name)
        ###print("var.name:",var.name)
        ###print("grad.op.name:",grad.op.name)
        ###print("var.op.name:",var.op.name)
        ##print("mask_name:",mask_name)
        ##print("mask:",mask)
        ##print("")
        mask_obj = tf.cast(mask,tf.float32)
        grad_m=tf.multiply(grad,mask_obj)
        clones_gradients[count]=(grad_m, var)
      count += 1
  return clones_gradients

def apply_pruning_to_grad(clones_gradients,pruningMask):
  """
  clones_gradients: [(<tf.Tensor 'gradients/AddN:0' shape=(1, 1, 4096, 5) dtype=float32>, <tf.Variable 'vgg_16/fc8/weights:0' shape=(1, 1, 4096, 5) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/fc8/BiasAdd_grad/tuple/control_dependency_1:0' shape=(5,) dtype=float32>, <tf.Variable 'vgg_16/fc8/biases:0' shape=(5,) dtype=float32_ref>), (<tf.Tensor 'gradients/AddN_1:0' shape=(1, 1, 4096, 4096) dtype=float32>, <tf.Variable 'vgg_16/fc7/weights:0' shape=(1, 1, 4096, 4096) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/fc7/BiasAdd_grad/tuple/control_dependency_1:0' shape=(4096,) dtype=float32>, <tf.Variable 'vgg_16/fc7/biases:0' shape=(4096,) dtype=float32_ref>), (<tf.Tensor 'gradients/AddN_2:0' shape=(3, 3, 3, 64) dtype=float32>, <tf.Variable 'vgg_16/conv1/conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>), (<tf.Tensor 'gradients/vgg_16/conv1/conv1_1/BiasAdd_grad/tuple/control_dependency_1:0' shape=(64,) dtype=float32>, <tf.Variable 'vgg_16/conv1/conv1_1/biases:0' shape=(64,) dtype=float32_ref>)]
  """
  ##print("######################apply_pruning_to_grad:###############################")
  if pruningMask is None:
    return clones_gradients
  for mask_name,mask in pruningMask:
    count = 0
    assign_ops=[]
    for grad,var in clones_gradients:
      if var.name == mask_name:
        ##print("grad.name:",grad.name)
        ##print("var.name:",var.name)
        ##print("grad.op.name:",grad.op.name)
        ##print("var.op.name:",var.op.name)
        ##print("mask_name:",mask_name)
        ##print("mask:",mask)
        ##print("")
        mask_obj = tf.cast(mask,tf.float32)
        ###
        ##print("mask_obj:",mask_obj)
        mask_DSD=FLAGS.pruning_gradient_update_ratio*(1-mask_obj)
        ##print("mask_DSD:",mask_DSD)
        mask_obj=tf.add(mask_obj,mask_DSD)
        ##print("mask_obj+mask_DSD:",mask_obj)
        ###
        grad_m=tf.multiply(grad,mask_obj)
        clones_gradients[count]=(grad_m, var)
      count += 1
  return clones_gradients


def get_pruning_mask(variables_to_pruning):
  """
  get pruning mask 
  My_variables_to_pruning: [<tf.Variable 'vgg_16/conv1/conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>, <tf.Variable 'vgg_16/conv1/conv1_1/biases:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'vgg_16/fc7/weights:0' shape=(1, 1, 4096, 4096) dtype=float32_ref>, <tf.Variable 'vgg_16/fc7/biases:0' shape=(4096,) dtype=float32_ref>]
  
  Return :
  return mask
  """    
  ##print("######################get_pruning_mask:###############################")
  if variables_to_pruning is None:
    return None
  mask=[]
  for W_B,rate in variables_to_pruning:
    var=W_B[0]
    shape=var.shape
    ##print("get_pruning_mask--var.name:",var.name) 
    ##print("get_pruning_mask--rate:",rate) 
    var_abs=tf.abs(var)
    var_vec=tf.reshape(var_abs,[-1])
    length=var_vec.shape[0].value
    top_k=tf.nn.top_k(var_vec,k=tf.cast(length*float(rate),tf.int32))
    thread=tf.reduce_min(top_k[0])
    thread_vec=tf.fill([length],thread) ##
    mask_vec=var_vec>thread_vec
    ##print("get_pruning_mask--mask_vec:",mask_vec)
    mask.append((var.name,tf.reshape(mask_vec,shape)))

    bias=W_B[1]
    if FLAGS.forbid_bias_bp :
      mask.append((bias.name,tf.fill(bias.shape,0)))
  return mask


def apply_pruning_to_var(variables_to_pruning,sess):
  """
  apply_pruning_to_var

  """    
  ##print("######################apply_pruning_to_var:###############################")
  if variables_to_pruning is None:
    return None
  pruningMask=[]
  for W_B,rate in variables_to_pruning:
    var=W_B[0]
    shape=var.shape
    print("var.name:",var.name) 
    print("rate=",float(rate))
    var_arr=sess.run(var)
    print("FLAGS.pruning_strategy:",FLAGS.pruning_strategy)
    ###print("var_arr_before_pruning:",var_arr)
    if FLAGS.pruning_strategy == "ABS":
      abs_var_arr=abs(var_arr)
      if float(rate) < 0.99 :
        print("sorting...")
        sort_abs_var_arr=np.sort(abs_var_arr,axis=None)
        ##int("sort_abs_var_arr.size=",sort_abs_var_arr.size)
        index=int(sort_abs_var_arr.size*(1-float(rate)))-1
      else:
        print("skip sort")
        index=-1
        continue
      print("index=",index)
      if index < 0:
        index = 0
      thread=sort_abs_var_arr[index]
      print("thread=",thread)
      print("abs_var_arr info:")
      print("std=",np.std(abs_var_arr))
      print("var=",np.var(abs_var_arr))
      print("mean=",np.mean(abs_var_arr))
      mask_arr=abs_var_arr<thread
      ###print("mask_arr=",mask_arr)
    elif FLAGS.pruning_strategy == "AXIS_0" :
      abs_var_arr=abs(var_arr)
      sort_abs_var_arr=np.sort(abs_var_arr,axis=0)
      ##int("sort_abs_var_arr.size=",sort_abs_var_arr.size)
      index=int(len(var_arr)*(1-float(rate)))-1
      print("index=",index)
      if index < 0:
        index = 0
      thread=np.zeros(var_arr.shape)
      for i in range(len(var_arr)):
        thread[i]=sort_abs_var_arr[index]
      print("thread=",thread)
      mask_arr=abs_var_arr<thread
      print("mask_arr=",mask_arr)
    elif FLAGS.pruning_strategy == "AUTO" :
      var_arr_reshape=var_arr.reshape([-1,var_arr.shape[-1]])
      var_arr_reshape_abs=abs(var_arr_reshape)
      var_arr_reshape_abs_sort=np.sort(var_arr_reshape_abs,axis=0)
      length=len(var_arr_reshape_abs_sort)
      index=int(length*(1-float(rate)))-1
      print("length=",length)
      print("var_arr_reshape_abs info:")
      print("std=",np.std(var_arr_reshape_abs))
      print("var=",np.var(var_arr_reshape_abs))
      print("mean=",np.mean(var_arr_reshape_abs))

      print("var_arr_reshape_abs masked 0 info:")
      var_arr_reshape_abs_m=np.ma.masked_values(var_arr_reshape_abs,0)
      print("std=",np.std(var_arr_reshape_abs_m))
      print("var=",np.var(var_arr_reshape_abs_m))
      print("mean=",np.mean(var_arr_reshape_abs_m))
      print("index=",index)
      if index < 0:
        index = 0
      thread_vec=var_arr_reshape_abs_sort[index]
      thread_arr=np.tile(thread_vec,[length,1])
      print("thread_vec.shape=",thread_vec.shape)
      print("thread_arr=",thread_arr)
      mask_arr=var_arr_reshape_abs<thread_arr
      mask_arr=mask_arr.reshape(var_arr.shape)
      print("mask_arr=",mask_arr)
      


    var_arr[mask_arr] = 0
    print("var_arr.shape=",var_arr.shape)
    ###print("var_arr_after_pruning:",var_arr)
    sess.run(var.assign(var_arr))
    pruningMask.append((var.name,mask_arr))
  return pruningMask


def get_variables_to_pruning():
  """Returns a list of variables to pruning.

  Returns:
    A list of variables to pruning.
  """

  if FLAGS.pruning_scopes is None:
    return None
  else:
    scopes = [scope.strip() for scope in FLAGS.pruning_scopes.split(',')]
  
  if FLAGS.pruning_rates:
    rates = [irate for irate in FLAGS.pruning_rates.split(',')]
  
  count=0
  for scope in scopes:
    if FLAGS.pruning_rates is None:
      rate=0.8
    else:
      rate=rates[count]
    scopes[count]=(scope,rate)
    count+=1

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
  
  variables_to_pruning = []
  for scope,rate in scopes:
    excluded = False
    for exclusion in exclusions:
      if scope == exclusion:
        excluded = True
        break

    if not excluded:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_pruning.append((variables,rate))
      
  return variables_to_pruning

###

def main(_):
  ###add for pruning
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)#add by lzlu  
  #sessGPU = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
  ##config = tf.ConfigProto()  
  ##config.gpu_options.allow_growth=True  
  ##sessGPU = tf.Session(config=config)  
  #sessGPU = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
  print("FLAGS.max_number_of_steps:",FLAGS.max_number_of_steps)
  print("FLAGS.learning_rate:",FLAGS.learning_rate)
  print("FLAGS.weight_decay:",FLAGS.weight_decay)
  print("FLAGS.batch_size:",FLAGS.batch_size)
  print("FLAGS.trainable_scopes:",FLAGS.trainable_scopes)
  print("FLAGS.pruning_rates:",FLAGS.pruning_rates)
  print("FLAGS.train_dir:",FLAGS.train_dir)
  print("FLAGS.checkpoint_path:",FLAGS.checkpoint_path)
  print("FLAGS.pruning_gradient_update_ratio:",FLAGS.pruning_gradient_update_ratio)
  ###
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)
    print("deploy_config.variables_device():")
    print(deploy_config.variables_device())
    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      with tf.device(deploy_config.inputs_device()):
        images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        tf.losses.softmax_cross_entropy(
            logits=end_points['AuxLogits'], onehot_labels=labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
      tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))
      ##add for pruning
      summaries.add(tf.summary.scalar('pruning_rate/' + variable.op.name,
                                      1-tf.nn.zero_fraction(variable)))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    print("deploy_config.optimizer_device():")
    print(deploy_config.optimizer_device())
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)

    ###add by lzlu
    ###variables = tf.model_variables()
    ###slim.model_analyzer.analyze_vars(variables, print_info=True)    
    ##print("variables_to_train:",variables_to_train)
    ##print("clones_gradients_before_pruning:",clones_gradients)
    variables_to_pruning=get_variables_to_pruning()
    pruningMask=get_pruning_mask(variables_to_pruning)
    ##print("pruningMask__grad:",pruningMask)
    ##print("My_variables_to_pruning__grad:",variables_to_pruning)
    clones_gradients=apply_pruning_to_grad(clones_gradients,pruningMask)
    ##print("clones_gradients_after_pruning:",clones_gradients)
    ##print("slim.get_model_variables():",slim.get_model_variables())
    ###
    
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)

    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ### add for pruning
    #######################
    # Config mySaver      #
    #######################
    class mySaver(tf.train.Saver):
      def restore(self,sess,save_path):
        ##print("mySaver--restore...!")
        tf.train.Saver.restore(self,sess,save_path)
        variables_to_pruning=get_variables_to_pruning()
        ##print("My_variables_to_pruning__restore:",variables_to_pruning)
        pruningMask=apply_pruning_to_var(variables_to_pruning,sess)
        ##print("mySaver--restore done!")
      def save(self,
               sess,
               save_path,
               global_step=None,
               latest_filename=None,
               meta_graph_suffix="meta",
               write_meta_graph=True,
               write_state=True):
        ##print("My Saver--save...!")
        tf.train.Saver.save(self,
                            sess,
                            save_path,
                            global_step,
                            latest_filename,
                            meta_graph_suffix,
                            write_meta_graph,
                            write_state)
        ##print("My Saver--save done!")

    saver=mySaver(max_to_keep=2)
    ###

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        saver=saver, #add for pruning
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()

