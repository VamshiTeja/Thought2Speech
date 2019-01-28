# -*- coding: utf-8 -*-
# @Author: vamshiteja
# @Date:   2017-11-05 06:09:28
# @Last Modified by:   vamshiteja
# @Last Modified time: 2017-11-21 17:07:22

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
    count = 0
    print (inputX.shape)
    for s in tf.split(inputX, num_or_size_splits=num_splits,axis=2):
        
        #1d convolution op
        s = tf.reshape(s, shape=(args.batch_size,50,args.num_channels))
        layer1_filter = tf.get_variable('layer1_filter', shape=(1, 62,  20), dtype=tf.float32)
        layer1_stride = 1
        layer2_filter = tf.get_variable('layer2_filter', shape=(3, 33, 1, 5), dtype=tf.float32)
        layer2_stride = [1, 1, 1, 1]
        layer3_filter = tf.get_variable('layer3_filter', shape=(11, 3, 5, 5), dtype=tf.float32)
        layer3_stride = [1, 1, 1, 1]

        print (s.shape)   
        layer1 = tf.nn.conv1d(s, layer1_filter, layer1_stride, padding='SAME')
        print(layer1.shape[2])
        layer1 = tf.layers.batch_normalization(layer1, training=args.isTraining)
        layer1 = tf.contrib.layers.dropout(layer1, keep_prob=args.keep_prob, is_training=args.isTraining)
        layer1 = tf.reshape(layer1, shape=(args.batch_size,int(layer1.shape[2]),int(layer1.shape[1])))
        layer1 = tf.expand_dims(layer1,-1)

        layer2 = tf.nn.conv2d(layer1, filter=layer2_filter, strides=layer2_stride, padding='SAME')
        layer2 = tf.layers.batch_normalization(layer2, training=args.isTraining)
        layer2 = tf.nn.max_pool(layer2, ksize=[1,2,5,1], strides=[1,2,5,1], padding='SAME')
        layer2 = tf.contrib.layers.dropout(layer2, keep_prob=args.keep_prob, is_training=args.isTraining)
        print(layer2)
        
        layer3 = tf.nn.conv2d(layer2, layer3_filter, layer3_stride, padding='SAME')
        layer3 = tf.layers.batch_normalization(layer3, training=args.isTraining)
        layer2 = tf.nn.max_pool(layer2, ksize=[1,2,5,1], strides=[1,2,5,1], padding='SAME')
        layer3 = tf.contrib.layers.dropout(layer3, keep_prob=args.keep_prob, is_training=args.isTraining)
        layer3 = tf.reshape(layer3, shape=(args.batch_size,1,int(layer3.shape[1]*layer3.shape[2]*layer3.shape[3])))
        print(layer3)
        if(count==0):
            Conv_feat = tf.Variable(layer3)
        else:
            Conv_feat = tf.concat((Conv_feat,layer3), axis=1)
        count = count + 1
    
    print(Conv_feat)
    # 1 recurrent layers
    # inputs must be [batch_size ,max Time, ...]
    layer4_cell = cell_fn(args.num_hidden, activation=args.activation)
    layer4 = tf.nn.dynamic_rnn(layer4_cell, Conv_feat, sequence_length=seqLengths, time_major=False) 
    layer4 = tf.layers.batch_normalization(layer4, training=args.isTraining)
    layer4 = tf.contrib.layers.dropout(layer4, keep_prob=args.keep_prob[3], is_training=args.is_training)

    # fully-connected layer
    layer_fc = tf.layers.dense(layer4, args.num_hidden_fc)

    return layer_fc


class DeepSpeech2(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        if args.layerNormalization is True:
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
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size}

            output_fc = build_deepSpeech2(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, output_fc, self.seqLengths))
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
            self.predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(output_fc, self.seqLengths, merge_repeated=False)[0][0])
            
            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            
            self.initial_op = tf.global_variables_initializer()
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)

