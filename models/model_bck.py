# -*- coding: utf-8 -*-
# @Author: vamshiteja
# @Date:   2017-11-05 06:09:28
# @Last Modified by:   vamshiteja
# @Last Modified time: 2017-12-10 20:12:16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import os 
import argparse
import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import setAttrs
from utils.utils import list_to_sparse_tensor
from utils.utils import dropout
from utils.utils import get_edit_distance

from utils.lnRNNCell import BasicRNNCell as lnBasicRNNCell
from utils.lnRNNCell import GRUCell as lnGRUCell
from utils.lnRNNCell import BasicLSTMCell as lnBasicLSTMCell

def conv_2d(input, kernel_shape,bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    conv =  tf.nn.relu(conv + biases)
    return conv

def conv_1d(input,kernel_shape,bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv1d(input, weights,
        stride=1, padding='SAME')
    conv =  tf.nn.relu(conv + biases)
    return conv

def eegnet(inputX,args,maxTimeSteps):
    with tf.variable_scope("eegnet") as eegnet:
        with tf.variable_scope("conv1d"):
            conv1d = conv_1d(inputX, kernel_shape=(1, 62,  20), bias_shape=(20))
            conv1d = tf.layers.batch_normalization(conv1d,training=args.isTraining)
            conv1d = tf.contrib.layers.dropout(conv1d, keep_prob=args.keep_prob,is_training=args.isTraining)
            conv1d = tf.reshape(conv1d, shape=(args.batch_size,int(conv1d.shape[2]),int(conv1d.shape[1])))
            conv1d = tf.expand_dims(conv1d,-1)
        with tf.variable_scope("conv2d_1"):
            conv2d_1 = conv_2d(conv1d, kernel_shape=(3, 33, 1, 5), bias_shape=[5])
            conv2d_1 = tf.layers.batch_normalization(conv2d_1,training=args.isTraining)
            conv2d_1 = tf.nn.max_pool(conv2d_1, ksize=[1,2,5,1], strides=[1,2,5,1], padding='SAME')
            conv2d_1 = tf.contrib.layers.dropout(conv2d_1, keep_prob=args.keep_prob,is_training=args.isTraining)
        with tf.variable_scope("con2d_2"):
            conv2d_2 = conv_2d(conv2d_1, kernel_shape=(11, 3, 5, 5), bias_shape=[5])
            conv2d_2 = tf.layers.batch_normalization(conv2d_1,training=args.isTraining)
            conv2d_2 = tf.nn.max_pool(conv2d_2, ksize=[1,2,5,1], strides=[1,2,5,1], padding='SAME')
            conv2d_2 = tf.contrib.layers.dropout(conv2d_2, keep_prob=args.keep_prob,is_training=args.isTraining)
            conv2d_2 = tf.reshape(conv2d_2, shape=(args.batch_size,1,int(conv2d_2.shape[1]*conv2d_2.shape[2]*conv2d_2.shape[3])))
    #eegnet.reuse_variables()
    return conv2d_2


def build_deepSpeech2(args,
                      maxTimeSteps,
                      inputX,
                      cell_fn,
                      seqLengths,
                      time_major=False):
    ''' Parameters:
          maxTimeSteps: maximum time steps of input spectrogram power
          inputX:  [batch, channels, time_len]
          seqLengths: lengths of samples in a mini-batch
    '''
    num_splits = int(maxTimeSteps/50)
    #count = 0
    print (inputX.shape)
    inputX_reshape  = tf.reshape(inputX, shape=(args.batch_size,maxTimeSteps,args.num_channels))
    inputX_splitted = tf.split(inputX_reshape, num_or_size_splits=num_splits, axis=1)

    
    conv_feat = tf.get_variable("conv_feat",shape=(args.batch_size,1,50))
    with tf.variable_scope("conv_feat") as scope:
        for i in range(num_splits):
            conv = (eegnet(inputX_splitted[i], args, maxTimeSteps))
            scope.reuse_variables()
            if(i==0):
                conv_feat = tf.assign(conv_feat,conv)
            else:
                conv_feat = tf.concat((conv_feat,conv), axis=1)

    #tf.get_variable_scope().reuse_variables()
    print(conv_feat)
    # 1 recurrent layers
    # inputs must be [batch_size ,max Time, ...]
    #layer4_cell = cell_fn(args.num_hidden, activation=args.activation)
    #print(layer4_cell)
    with tf.variable_scope("cell_def"):
        layer4_cell = tf.contrib.rnn.LSTMCell(args.num_hidden)
    #tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("rnn_def"):
        layer4,_ = tf.nn.dynamic_rnn(layer4_cell, conv_feat, sequence_length=seqLengths, time_major=False,dtype=tf.float32) 
    #tf.get_variable_scope().reuse_variables()

    layer4 = tf.layers.batch_normalization(layer4, training=args.isTraining)
    layer4 = tf.contrib.layers.dropout(layer4, keep_prob=0.5, is_training=args.isTraining)
    print(layer4)
    # fully-connected layer
    layer_fc = tf.layers.dense(layer4, args.num_hidden_fc)
    return layer_fc

class DeepSpeech2(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        if args.layerNormalization  is True:
            if args.rnncell == 'rnn':
                self.cell_fn = lnBasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = lnGRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = lnBasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))
        else:
            if args.rnncell == 'rnn':
                self.cell_fn = rnn_cell.BasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = tf.contrib.rnn.GRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = core_rnn_cell_impl.BasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))

    @describe
    def build_graph(self, args, maxTimeSteps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # according to DeepSpeech2 paper, input is the spectrogram power of audio, but if you like,
            # you can also use mfcc feature as the input.
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(args.batch_size, args.num_channels,maxTimeSteps))  
            #inputXrs = tf.reshape(self.inputX, [args.batch_size, args.num_channels, maxTimeSteps])
            #self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]

            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))
            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_hidden_fc': args.num_hidden_fc,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch_size': args.batch_size}

            output_fc = build_deepSpeech2(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)
            print(self.targetY)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, output_fc, self.seqLengths,time_major=False))
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()

            if args.grad_clip == -1:
                # not apply gradient clipping
                self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            else:
                # apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
                opti = tf.train.AdamOptimizer(args.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
            num_splits = tf.to_int32(self.maxTimeSteps/50)
            output = tf.reshape(output_fc,shape=(num_splits,self.config['batch_size'],self.config['num_hidden_fc']))
            self.predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(output, self.seqLengths, merge_repeated=False)[0][0])
            print(self.predictions)
            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            
            self.initial_op = tf.initialize_all_variables()
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)

