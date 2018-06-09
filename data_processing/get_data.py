import q_common as qc
import pycnbi_utils as pu
from mne import Epochs, pick_types
import os, sys
import multiprocessing as mp
import traceback

DATADIR = r'E:\data_IRMCT\eeg_meg_data\data20180607\Records\fif\10STASK'

def get_data(DATADIR, cfg):
    ftrain= []
    for f in qc.get_file_list(DATADIR, fullpath=True):  
#         print(f)
        f=f[0] #f if list getf[0]is filename
#         print(f)
        if f[-4:] in ['.fif','.fiff','.pcl','.bdf','.gdf']:
            ftrain.append(f)
    print ('training files including: ',ftrain)


    spfilter= cfg.SP_FILTER
    tpfilter= cfg.TP_FILTER

    # Load multiple files
    multiplier= 1
    raw, events= pu.load_multi(ftrain, spfilter=spfilter, multiplier=multiplier)
    #print(raw._data.shape)  #(17L, 2457888L)
    triggers= { cfg.tdef.by_value[c]:c for c in set(cfg.TRIGGER_DEF) }

    # Pick channels
    if cfg.CHANNEL_PICKS is None:
        picks= pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads') 
        #print (picks) # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    else:
        picks= []
        for c in cfg.CHANNEL_PICKS:
            if type(c)==int:
                picks.append(c)
            elif type(c)==str:
                picks.append( raw.ch_names.index(c) )
            else:
                raise RuntimeError, 'CHANNEL_PICKS is unknown format.\nCHANNEL_PICKS=%s'% cfg.CHANNEL_PICKS
 
    if max(picks) > len(raw.info['ch_names']):
        print('ERROR: "picks" has a channel index %d while there are only %d channels.'%\
            ( max(picks),len(raw.info['ch_names']) ) )
        sys.exit(-1)
# 
    # Spatial filter
    if cfg.SP_CHANNELS is None:
        spchannels= pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    else:
        spchannels= []
        for c in cfg.SP_CHANNELS:
            if type(c)==int:
                spchannels.append(c)
            elif type(c)==str:
                spchannels.append( raw.ch_names.index(c) )
            else:
                raise RuntimeError, 'SP_CHANNELS is unknown format.\nSP_CHANNELS=%s'% cfg.SP_CHANNELS
# 
    # Spectral filter
    if tpfilter is not None:
        raw= raw.filter( tpfilter[0], tpfilter[1], picks=picks, n_jobs= mp.cpu_count() )
    if cfg.NOTCH_FILTER is not None:
        raw= raw.notch_filter( cfg.NOTCH_FILTER, picks=picks, n_jobs= mp.cpu_count() )
    
    # Read epochs
    try:
        
        epochs_train= Epochs(raw, events, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1], proj=False,\
            picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False, detrend=None)
        #print (epochs_train)# <Epochs  |  n_events : 422 (all good), tmin : 1.0 (s), tmax : 2.0 (s), baseline : None, ~26.5 MB, data loaded,'LEFT_GO': 212, 'RIGHT_GO': 210>
    except:
        print('\n*** (trainer.py) ERROR OCCURRED WHILE EPOCHING ***\n')
        traceback.print_exc()
    epochs_data= epochs_train.get_data()
    return epochs_train, epochs_data



