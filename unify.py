# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2017-10-05 14:06:35
# @Last Modified by:   vamshi
# @Last Modified time: 2017-10-05 15:55:03


import os
import sys
import shutil

folders = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14','MM15', 
			'MM16', 'MM18', 'MM19', 'MM20', 'MM21','P02'] 

root = os.getcwd()
destination = "./unify" 

for folder in folders:
	epoch_dir = os.path.join(root,folder)
	epoch_file = os.path.join(epoch_dir,'epoch_data_simple.mat')

	label_dir = os.path.join(epoch_dir,'kinect_data')
	label_file = os.path.join(label_dir,folder+'_p.txt')
	shutil.copy2(label_file, destination)
	#os.rename(epoch_file,destination+"/epoch_data_" + folder + ".mat")
	print epoch_dir

