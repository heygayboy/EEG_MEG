from get_data import get_data
from trainer import get_psd_feature, crossval_epochs
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import multiprocessing as mp
import numpy as np
import sys
import sklearn.metrics as skmetrics
from sklearn.model_selection import StratifiedShuffleSplit
import q_common as qc

DATADIR_easy = r'E:\data_IRMCT\eeg_meg_data\data20180607\Records\fif\10STASK'
DATADIR_heavy = r'E:\data_IRMCT\eeg_meg_data\data20180607\Records\fif\5STASK'

def load_config():
    import imp  
    cfg_module  = "config.py"
    if cfg_module[-3:]=='.py':
        cfg_module= cfg_module[:-3]
    cfg= imp.load_source(cfg_module, "./config.py")
    return cfg

def init_cls(cfg):
    # Init a classifier
    if cfg.CLASSIFIER=='RF':
        # Make sure to set n_jobs=cpu_count() for training and n_jobs=1 for testing.
        cls= RandomForestClassifier(n_estimators=cfg.RF['trees'], max_features='auto',\
            max_depth=cfg.RF['maxdepth'], n_jobs=mp.cpu_count(), class_weight='balanced' )
    elif cfg.CLASSIFIER=='LDA':
        cls= LDA()
#     elif cfg.CLASSIFIER=='rLDA':
#         cls= rLDA(cfg.RLDA_REGULARIZE_COEFF)
    else:
        raise RuntimeError, '*** Unknown classifier %s'% cfg.CLASSIFIER
    return cls

def cross_validation(X_data, Y_data,cfg):
    Y_data = Y_data.flatten()
    cv= StratifiedShuffleSplit(cfg.CV_FOLDS, test_size=cfg.CV_TEST_RATIO, random_state=0)
    scores= []
    cnum = 1
    timer= qc.Timer()
    for train_index,test_index in cv.split(X_data, Y_data):
        timer.reset()
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
        classifier.fit( X_train, y_train )
        Y_pred= classifier.predict( X_test )
        score= skmetrics.accuracy_score(y_test, Y_pred)
        scores.append( score )
        print('Cross-validation %d / %d (%.2f) - %.1f sec'% (cnum, cfg.CV_FOLDS, score,timer.sec()) )
        cnum += 1
    print('Average accuracy: %.2f'% np.mean(scores) )
    
if __name__=='__main__':
    cfg = load_config()
    epochs_easy, data_easy = get_data(DATADIR_easy, cfg)
    print ('data_easy.shape', data_easy.shape)
    epochs_heavy, data_heavy = get_data(DATADIR_heavy, cfg)
#     print (data_heavy.shape)
    data_heavy = data_heavy[-11:-1,:,:]
    print ('data_heavy.shape', data_heavy.shape)
    
    res_easy= get_psd_feature(epochs_easy, cfg.EPOCH, cfg.PSD, feat_picks = None)
    res_heavy = get_psd_feature(epochs_heavy, cfg.EPOCH, cfg.PSD, feat_picks = None)
    
    psd_easy = res_easy['X_data']
    print ('psd_easy.shape', psd_easy.shape)
    psd_heavy = res_heavy['X_data'][-11:-1,:,:]
    print ('psd_heavy.shape', psd_heavy.shape)
    
    '''
    classify
    '''
    classifier = init_cls(cfg)
    X_data = np.concatenate((psd_easy,psd_heavy),axis=0)
    label_easy = np.zeros((psd_easy.shape[0], 1))
    label_heavy = np.ones((psd_heavy.shape[0], 1))
    Y_data = np.concatenate((label_easy, label_heavy), axis= 0)
    print('X_data.shape', X_data.shape)
    print('Y_data.shape', Y_data.shape)
    cross_validation(X_data, Y_data,cfg)
    
#     x_data = X_data.reshape(X_data.shape[0], X_data.shape[1]*X_data.shape[2])
#     classifier.fit( x_data, Y_data )
#     Y_pred= classifier.predict( x_data )
#     score= skmetrics.accuracy_score(Y_data, Y_pred)
#     print(score)
    

    
 
       
    