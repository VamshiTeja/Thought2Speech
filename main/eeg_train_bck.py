# -*- coding: utf-8 -*-
# @Author: vamshiteja
# @Date:   2017-10-29 18:59:48
# @Last Modified by:   vamshiteja
# @Last Modified time: 2017-12-25 19:15:32
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import time
import datetime
import os
from six.moves import cPickle
from functools import wraps
import random
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn

from utils.utils import load_batched_data
from utils.utils import describe
from utils.utils import getAttrs
from utils.utils import output_to_sequence
from utils.utils import list_dirs
from utils.utils import logging
from utils.utils import count_params
from utils.utils import target2phoneme
from utils.utils import get_edit_distance

from utils.taskUtils import get_num_classes
from utils.taskUtils import check_path_exists
from utils.taskUtils import dotdict
from utils.functionDictUtils import model_functions_dict
from utils.functionDictUtils import activation_functions_dict
from utils.functionDictUtils import optimizer_functions_dict

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

task = "eeg"

train_dataset = 'kara-one'
test_dataset = 'kara-one'

level = 'cha'
model_fn = model_functions_dict['deepSpeech2']
rnncell = 'lstm'
num_layer = 2

activation_fn = activation_functions_dict['tanh']
optimizer_fn = optimizer_functions_dict['adam']

batch_size = 2
num_hidden = 128
num_hidden_fc = 13
num_channels = 62
num_classes = 13
num_epochs = 100
lr = 0.0001
grad_clip = 1
datadir = '/home/vamshi/Projects/Thought2Speech/T2S/data/'
logdir = '/home/vamshi/Projects/Thought2Speech/T2S/logdir'

savedir = os.path.join(logdir, level, 'save')
resultdir = os.path.join(logdir, level, 'result')
loggingdir = os.path.join(logdir, level, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

mode = 'train'
keep = True
keep_prob = 1-0.5
if(mode=='train'):
	isTraining = True
else:
	isTraining = False

print('%s mode...'%str(mode))
if mode == 'test' or mode == 'dev':
  batch_size = 1
  num_epochs = 1

train_dataset = "../data/train/train.pickle"
test_dataset  = "../data/test/test.pickle"

def get_data(level, train_dataset, test_dataset, mode):
	if mode == 'train':
		with open(train_dataset,'rb') as f:
			save = pickle.load(f)
			X_train = save['X_train']
			y_train = save['labels']
		return X_train, y_train

	if mode == 'test':
		with open(test_dataset,'rb') as f:
			save = pickle.load(f)
			X_test = save['X_train']
			y_test = save['labels']
		return X_test, y_test

logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(), 
	'%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))


class Runner(object):

	def _default_configs(self):
	  return {'level': level,
			  'rnncell': rnncell,
			  'batch_size': batch_size,
			  'num_hidden': num_hidden,
			  'num_hidden_fc': num_hidden_fc,
			  'num_channels': num_channels,
			  'num_class': num_classes,
			  'num_layer': num_layer,
			  'layerNormalization' : False,
			  'activation': activation_fn,
			  'optimizer': optimizer_fn,
			  'isTraining' : isTraining,
			  'learning_rate': lr,
			  'keep_prob': keep_prob,
			  'grad_clip': grad_clip,
			}

	@describe
	def load_data(self, X, labels, batchSize, mode, level):
		return load_batched_data(X, labels, batchSize, mode, level)

	def run(self):
		# load data
		args_dict = self._default_configs()
		args = dotdict(args_dict)
		
		X, labels = get_data(level, train_dataset, test_dataset, mode)
		print("X :",len(X))

		batchedData, maxTimeSteps, totalN = self.load_data(X,labels,batch_size,mode,level)
		model = model_fn(args, maxTimeSteps)
		model.build_graph(args,maxTimeSteps)
		#print("hello")
		#num_params = count_params(model, mode='trainable')
		#all_num_params = count_params(model, mode='all')
		#model.config['trainable params'] = num_params
		#model.config['all params'] = all_num_params
		print(model.config)

		with tf.Session(graph=model.graph) as sess:
			# restore from stored model
			if keep == True:
				ckpt = tf.train.get_checkpoint_state(savedir)
				if ckpt and ckpt.model_checkpoint_path:
					model.saver.restore(sess, ckpt.model_checkpoint_path)
					print('Model restored from:' + savedir)
				else:
					sess.run(model.initial_op)
			else:
				print('Initializing')
				sess.run(model.initial_op)
		
			if(mode=='train'):
				for epoch in range(num_epochs):
					# training
					start = time.time()
					if mode == 'train':
						print('Epoch {} ...'.format(epoch + 1))

					batchErrors = np.zeros(len(batchedData))
					batchRandIxs = np.random.permutation(len(batchedData))
					for batch, batchOrigI in enumerate(batchRandIxs):
						print(batch)
						batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
									
						batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
						feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs,
									model.targetVals: batchTargetVals, model.targetShape: batchTargetShape,
									model.seqLengths: batchSeqLengths}
						print(batchSeqLengths)
						if level == 'cha':
							if mode == 'train':
								_, l, pre, y, er = sess.run([model.optimizer, model.loss,
									model.predictions, model.targetY, model.errorRate],
									feed_dict=feedDict)

								batchErrors[batch] = er
								print('\n{} mode, batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(
									level, batch+1, len(batchRandIxs), epoch+1, num_epochs, l, er/batch_size))


						if (batch+1) % 10 == 0:
							print('Truth:\n' + output_to_sequence(y, type='phn'))
							print('Output:\n' + output_to_sequence(pre, type='phn'))

						if ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (
							epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
							checkpoint_path = os.path.join(savedir, 'model.ckpt')
							model.saver.save(sess, checkpoint_path, global_step=epoch)
							print('Model has been saved in {}'.format(savedir))
						
					end = time.time()
					delta_time = end - start
					print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

					
					if (epoch + 1) % 1 == 0:
						checkpoint_path = os.path.join(savedir, 'model.ckpt')
						model.saver.save(sess, checkpoint_path, global_step=epoch)
						print('Model has been saved in {}'.format(savedir))
					epochER = batchErrors.sum() / totalN
					print('Epoch', epoch + 1, 'mean train error rate:', epochER)
					logging(model, logfile, epochER, epoch, delta_time, mode='config')
					logging(model, logfile, epochER, epoch, delta_time, mode=mode)

			elif(mode=='test'):

				_, l, pre, y, er = sess.run([model.optimizer, model.loss,
					model.predictions, model.targetY, model.errorRate],
					feed_dict=feedDict)
				with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
					result.write(output_to_sequence(y, type='phn') + '\n')
					result.write(output_to_sequence(pre, type='phn') + '\n')
					result.write('\n')
					epochER = batchErrors.sum() / totalN
					print(' test error rate:', epochER)
					logging(model, logfile, epochER, mode=mode)


if __name__ == '__main__':
  runner = Runner()
  runner.run()
