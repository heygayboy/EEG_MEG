# -*- coding: utf-8 -*-
"""
Settings for trainer.py

Kyuhwa Lee, 2015
"""

import pycnbi_config
import numpy as np

'''"""""""""""""""""""""""""""
 DATA
"""""""""""""""""""""""""""'''

#DATADIR= r'E:\MI_data\data2017-4\data\wjy4.13\Records\fif' # 'Trig1'
DATADIR= r'E:\MI_data\data2017-4\data\xhm5.17\Records\fif'

'''"""""""""""""""""""""""""""
Parameters for computing PSD
Ignored if LOAD_PSD == Ture

wlen: window length in seconds
wstep: window step (32 is enough for 512 Hz, or 256 for 2KHz)

"""""""""""""""""""""""""""'''

PSD= dict(fmin=4, fmax=30, wlen=1, wstep= 128)

'''"""""""""""""""""""""""""""
 EVENTS

 TRIGGER_DEF is ignored if LOAD_PSD==True
"""""""""""""""""""""""""""'''
# None or events filename (hardware events in raw file will be ignored)
# TODO: set this flag in load_multi to concatenate frame numbers in multiple files.


from triggerdef_16 import TriggerDef
tdef= TriggerDef()

TRIGGER_DEF= {tdef.LEFT_GO, tdef.RIGHT_GO}

EPOCH= [1, 5] 

# change WALk_GO event values
DEBUG_STAND_TRIGGERS= False


'''"""""""""""""""""""""""""""
 CHANNELS

 Pick a subset of channels for PSD. Note that Python uses zero-based indexing.
 However, for fif files saved using PyCNBI library, index 0 is the trigger channel
 and data channels start from index 1. (to be consistent with MATLAB)

 Ignored if LOAD_PSD= True

"""""""""""""""""""""""""""'''
CHANNEL_PICKS= None # use all channels

#CHANNEL_PICKS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
#CHANNEL_PICKS = [4,5,6,8,9,10,12,13,19,21,22,23,25,26,27,31,32]


'''"""""""""""""""""""""""""""
 FILTERS
"""""""""""""""""""""""""""'''
# apply spatial filter immediately after loading data
# SP_FILTER= None | 'car' | 'laplacian'
SP_FILTER= 'car'

# only consider the following channels while computing
SP_CHANNELS= CHANNEL_PICKS
SP_CHANNELS_Laplaian16={1:[2,5], 2:[1,3,6],3:[2,4,7],4:[3,8],5:[1,6,9],6:[2,5,7,10],7:[3,6,8,11],8:[4,7,12],9:[5,10,13],10:[6,9,11,14],\
                11:[7,10,12,15],12:[8,11,16],13:[9,14],14:[10,13,15],15:[11,14,16],16:[12,15]}


TP_FILTER=[1, 50]
#TP_FILTER= [1, 50]      frequency---low band filter

NOTCH_FILTER= None # None or list of values

'''"""""""""""""""""""""""""""
 FEATURE TYPE
"""""""""""""""""""""""""""'''

EXPORT_GOOD_FEATURES= True
FEAT_TOPN= 15 # show only the top N features

# Wavelet parameters
#DWT= dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 18])
DWT= dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30])
# export wavelets into MATLAB file
EXPORT_DWT= False


'''"""""""""""""""""""""""""""
 TimeLag parameters

 w_frames: window length in frames (samples) of downsampled data
 wstep: window step in downsampled data
 downsample: average every N-sample block (reduced to 1/N samples)
"""""""""""""""""""""""""""'''
TIMELAG= dict(w_frames=10, wstep=5, downsample=100)


'''"""""""""""""""""""""""""""
 CLASSIFIER
"""""""""""""""""""""""""""'''
# clasifier
#CLASSIFIER= 'RF' # RF | LDA | rLDA
CLASSIFIER = 'RF'
#CLASSIFIER= 'LDA'


# RF parameters
RF= dict(trees=400, maxdepth=100)

# rLDA parameter
RLDA_REGULARIZE_COEFF= 0.3


'''"""""""""""""""""""""""""""
 CROSS-VALIDATION & TESTING
"""""""""""""""""""""""""""'''
# do cross-validation?
CV_PERFORM= 'StratifiedShuffleSplit'  # 'StratifiedShuffleSplit' | 'LeaveOneOut' | None
CV_TEST_RATIO= 0.2 # ignored if LeaveOneOut
CV_FOLDS= 10

# testing file
ftest= ''
#ftest=r'G:\data\Records\fif\20170520-092212-raw.fif'
#ftest = r'E:\LP_magnien\fif\20160511-093524-raw.fif'

'''"""""""""""""""""""""""""""
 ETC
"""""""""""""""""""""""""""'''
# write to log file?
USE_LOG= False




