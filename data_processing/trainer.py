# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os, sys, timeit, platform
import pdb
import q_common as qc
import pycnbi_utils as pu
import mne
from mne import Epochs, pick_types
from sklearn.learning_curve import learning_curve
import multiprocessing as mp
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc 
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
#from rlda import rLDA
from mpl_toolkits.mplot3d import Axes3D


def crossval_epochs(cv, epochs_data, labels, cls, label_names=None, do_balance=False):
    """
    Epoch (trial) based cross-validation

    cv: scikit-learn cross-validation object
    epochs_data: np.array of [epochs x samples x features]
    cls: classifier
    labels: vector of integer labels
    label_names: associated label names {0:'Left', 1:'Right', ...}
    do_balance: oversample or undersample to match the number of samples among classes
    """

    scores= []
    cnum= 1
    timer= qc.Timer()
    label_set= np.unique(labels)
    num_labels= len(label_set)
    cm= np.zeros( (num_labels, num_labels) )
    if label_names==None:
        label_names= {l:'%s'%l for l in label_set}

    # select train and test trial ID's
    for i, (train, test) in enumerate(cv): 
        timer.reset()
        X_train= np.concatenate( epochs_data[train] )
        X_test= np.concatenate( epochs_data[test] )
        Y_train= np.concatenate( labels[train] )
        Y_test= np.concatenate( labels[test] )

        cls.n_jobs= mp.cpu_count()
        cls.fit( X_train, Y_train )
        cls.n_jobs= 1
        #score= cls.score( X_test, Y_test )
        Y_pred= cls.predict( X_test )
        score= skmetrics.accuracy_score(Y_test, Y_pred)
        cm += skmetrics.confusion_matrix(Y_test, Y_pred, label_set)
        scores.append( score )
        print('Cross-validation %d / %d (%.2f) - %.1f sec'% (cnum, len(cv), score,timer.sec()) )
        cnum += 1
        '''
        #ROC AUC
        if(num_labels==2):  
            mean_tpr = 0.0  
            mean_fpr = np.linspace(0, 1, 100)  
            all_tpr = [] 
            probas_ = cls.fit(X_train, Y_train ).predict_proba(X_test) 
            Y_test[np.where(Y_test==label_set[0])]=0
            Y_test[np.where(Y_test==label_set[1])]=1
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1]) 
            mean_tpr += interp(mean_fpr, fpr, tpr) 
            print (mean_tpr)
            mean_tpr[0] = 0.0 
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc)) 
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
   
    #mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
    print (mean_tpr)
    mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
    #画平均ROC曲线  
    #print mean_fpr,len(mean_fpr)  
    #print mean_tpr  
    plt.plot(mean_fpr, mean_tpr, 'k--',  
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
       
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating characteristic example')  
    plt.legend(loc="lower right")  
    plt.show()
    '''    
        
        
    # show confusion matrix
    cm_rate= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('\nY: ground-truth, X: predicted')
    for l in label_set:
        print('%-5s'% label_names[l][:5], end='\t')
    print()
    for r in cm_rate:
        for c in r:
            print('%-5.2f'% c, end='\t')
        print()
    print('Average accuracy: %.2f'% np.mean(scores) )

    '''
    # plot confusion matrix
    plt.matshow(cm_rate)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    '''
    


    return np.array(scores)

def get_psd_feature(epochs_train, window, psdparam, feat_picks=None):
    """
    params:
      epochs_train: mne.Epochs object or list of mne.Epochs object.
      window: time window range for computing PSD. Can be [a,b] or [ [a1,b1], [a1,b2], ...]
    """

    if type(window[0]) is list:
        sfreq= epochs_train[0].info['sfreq']
        wlen= []
        w_frames= []
        # multiple PSD estimators, defined for each epoch
        if type(psdparam) is list:
            print('MULTIPLE PSD FUNCTION NOT IMPLEMENTED YET.')
            sys.exit(-1)

            '''
            TODO: implement multi-PSD for each epoch
            '''
            assert len(psdparam)==len(window)
            for i, p in enumerate(psdparam):
                if p['wlen']==None:
                    wl= window[i][1] - window[i][0]
                else:
                    wl= p['wlen']
                wlen.append( wl )
                w_frames.append( int(sfreq * wl) )
        # same PSD estimator for all epochs
        else:
            for i, e in enumerate(window):
                if psdparam['wlen']==None:
                    wl= window[i][1] - window[i][0]
                else:
                    wl= psdparam['wlen']
                assert wl > 0
                wlen.append( wl )
                w_frames.append( int(sfreq * wl) )
    else:
        sfreq= epochs_train.info['sfreq']
        wlen= window[1] - window[0]
        if psdparam['wlen'] is None:
            psdparam['wlen']= wlen
        w_frames= int(sfreq * psdparam['wlen']) # window length

    psde= mne.decoding.PSDEstimator(sfreq=sfreq, fmin=psdparam['fmin'],\
        fmax=psdparam['fmax'], bandwidth=None, adaptive=False, low_bias=True,\
        n_jobs=1, normalization='length', verbose=None)

    print('\n>> Computing PSD for training set')
    if type(epochs_train) is list:
        X_all= []
        for i, ep in enumerate(epochs_train):
            X, Y_data= pu.get_psd(ep, psde, w_frames[i], psdparam['wstep'], feat_picks)
            X_all.append(X)
        # concatenate along the feature dimension
        X_data= np.concatenate( X_all, axis=2 )
    else:
        X_data, Y_data= pu.get_psd(epochs_train, psde, w_frames, psdparam['wstep'], feat_picks)

    # return a class-like data structure
    return dict(X_data= X_data, Y_data= Y_data, wlen= wlen, w_frames= w_frames, psde= psde)


def pre_xdata(xdata, k=1):
    scaler = StandardScaler()
    print('xdata', xdata.shape)
    xdata = np.squeeze(xdata)
    print('xdata', xdata.shape)
    scaler.fit(xdata)
    x_train = scaler.transform(xdata)
    model = pca.PCA(n_components=k).fit(x_train)   # 拟合数据，n_components定义要降的维度
#     print('model', model)
    Z = model.transform(x_train)    # transform就会执行降维操作
    print('Z', Z.shape)
#     Ureduce = model.components_     # 得到降维用的Ureduce
#     print('Ureduce', Ureduce.shape)
#     x_rec = np.dot(Z,Ureduce)       # 数据恢复
#     print('x_rec', x_rec.shape)
    Z = Z[:, np.newaxis,:]
    print('Z', Z.shape)
    return Z

def plot_pca_componet(X_data, Y_data):
    '''
    plt using PCA to jiangwei 2--D feature and plot
    '''
    X_data_1D = np.squeeze(pre_xdata(X_data, k=3))
    y_data_1D = np.squeeze(Y_data)
    print('X_data_1D', X_data_1D.shape)
    print('Y_data', Y_data.shape) 
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    idx_1 = np.where(y_data_1D == 9)
    ax.scatter(X_data_1D[idx_1,0], X_data_1D[idx_1,1], X_data_1D[idx_1,2], marker = 'x', c = 'y', label='left', linewidths = 8) 
    idx_2 = np.where(y_data_1D == 11)
    ax.scatter(X_data_1D[idx_2,0], X_data_1D[idx_2,1], X_data_1D[idx_2,2],marker = '+', c = 'r', label='right', linewidths = 8)  
    ax.set_xlabel("feature component 1") 
    ax.set_ylabel("feature component 2")
    ax.set_zlabel("feature component 3")
    ax.set_title('Scatter Plot') 
    plt.legend()
    plt.show() 
    
def run_trainer(cfg, ftrain, interactive=False):
    # feature selection?
    datadir= cfg.DATADIR
    feat_picks= None
    txt= 'all'

    do_balance= False

    # preprocessing, epoching and PSD computation
    n_epochs= {}

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
        if interactive:
            print('Dropping into a shell.\n')
            pdb.set_trace()
        raise RuntimeError
    '''
    epochs_data= epochs_train.get_data()
    print (epochs_data.shape)  #(422L, 16L, 513L)  trail*channel*caiyangdian
    
    #Visualize raw data for some channel in some trial
    ptrial=1
    trail=np.zeros((len(spchannels),epochs_data.shape[2]))
    print(trail)
    for pch in range(len(spchannels)):
        print(pch)
        trail[pch,::] =epochs_data[ptrial,pch,::]
    color=["b","g","r",'c','m','y','k','w',"b","g","r",'c','m','y','k','w']
    linstyle=['-','-','-','-','-','-','-','-','--','--','--','--','--','--','--','--',]
    for pch in range(len(spchannels)):
        print(color[pch])
        print(linstyle[pch])
        plt.plot(np.linspace(cfg.EPOCH[0], cfg.EPOCH[1], epochs_data.shape[2]), trail[pch,::],c=color[pch],ls=linstyle[pch],
                 label='channel %d'%(pch+1),lw=0.5)  
        
    plt.xlabel('time/s')  
    plt.ylabel('voltage/uV')  
    plt.title('Viewer')  
    plt.legend(loc="lower right")  
    plt.show()
    '''
    
    
    label_set= np.unique(triggers.values())
    sfreq= raw.info['sfreq']
  
    # Compute features
    res= get_psd_feature(epochs_train, cfg.EPOCH, cfg.PSD, feat_picks)
    X_data= res['X_data'] 
    Y_data= res['Y_data']
    wlen= res['wlen']
    w_frames= res['w_frames']
    psde= res['psde']
    psdfile= '%s/psd/psd-train.pcl'% datadir
    plot_pca_componet(X_data, Y_data)
    
    
    
  
    psdparams= cfg.PSD
#     print (events)
    for ev in triggers:
        print (ev) 
        n_epochs[ev]= len( np.where(events[:,-1]==triggers[ev])[0] )#{'RIGHT_GO': 150, 'LEFT_GO': 150} total trails
  
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
  
    # Cross-validation
    if cfg.CV_PERFORM is not None:
        ntrials, nsamples, fsize= X_data.shape
  
        if cfg.CV_PERFORM=='LeaveOneOut':
            print('\n>> %d-fold leave-one-out cross-validation'% ntrials)
            cv= LeaveOneOut(len(Y_data))
        elif cfg.CV_PERFORM=='StratifiedShuffleSplit':
            print('\n>> %d-fold stratified cross-validation with test set ratio %.2f'% (cfg.CV_FOLDS, cfg.CV_TEST_RATIO))
            cv= StratifiedShuffleSplit(Y_data[:,0], cfg.CV_FOLDS, test_size=cfg.CV_TEST_RATIO, random_state=0)
        else:
            print('>> ERROR: Unsupported CV method yet.')
            sys.exit(-1)
        print('%d trials, %d samples per trial, %d feature dimension'% (ntrials, nsamples, fsize) )
  
        # Do it!
        scores= crossval_epochs(cv, X_data, Y_data, cls, cfg.tdef.by_value, do_balance)
         
         
        '''
        #learning curve        
        train_sizes,train_loss,test_loss=learning_curve(cls,X_data.reshape(X_data.shape[0]*X_data.shape[1],X_data.shape[2]),Y_data.reshape(Y_data.shape[0]*Y_data.shape[1]),train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        print(X_data.shape)
        print(Y_data.shape)
        train_loss_mean=np.mean(train_loss,axis=1)
        test_loss_mean=np.mean(test_loss,axis=1)
        plt.plot(train_sizes,train_loss_mean,label='training')
        plt.plot(train_sizes,test_loss_mean,label='Cross-validation')
        plt.xlabel('training examples')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.show()  
        ''' 
           
 
        # Results
        print('\n>> Class information')
        for ev in np.unique(Y_data):
            print('%s: %d trials'% (cfg.tdef.by_value[ev], len(np.where(Y_data[:,0]==ev)[0])) )
        if do_balance:
            print('The number of samples was balanced across classes. Method:', do_balance)
  
        print('\n>> Experiment conditions')
        print('Spatial filter: %s (channels: %s)'% (spfilter, spchannels) )
        print('Spectral filter: %s'% tpfilter)
        print('Notch filter: %s'% cfg.NOTCH_FILTER)
        print('Channels: %s'% picks)
        print('PSD range: %.1f - %.1f Hz'% (psdparams['fmin'], psdparams['fmax']) )
        print('Window step: %.1f msec'% (1000.0 * psdparams['wstep'] / sfreq) )
        if type(wlen) is list:
            for i, w in enumerate(wlen):
                print('Window size: %.1f sec'% (w) )
                print('Epoch range: %s sec'% (cfg.EPOCH[i]))
        else:
            print('Window size: %.1f sec'% (psdparams['wlen']) )
            print('Epoch range: %s sec'% (cfg.EPOCH))
  
        #chance= 1.0 / len(np.unique(Y_data))
        cv_mean, cv_std= np.mean(scores), np.std(scores)
        print('\n>> Average CV accuracy over %d epochs'% ntrials)
        if cfg.CV_PERFORM in ['LeaveOneOut','StratifiedShuffleSplit']:
            print("mean %.3f, std: %.3f" % (cv_mean, cv_std) )
        print('Classifier: %s'% cfg.CLASSIFIER)
        if cfg.CLASSIFIER=='RF':
            print('            %d trees, %d max depth'% (cfg.RF['trees'], cfg.RF['maxdepth']) )
  
        if cfg.USE_LOG:
            logfile= '%s/result_%s_%s.txt'% (datadir, cfg.CLASSIFIER, txt)
            logout= open(logfile, 'a')
            logout.write('%s\t%.3f\t%.3f\n'% (ftrain[0], np.mean(scores), np.var(scores)) )
            logout.close()
  
    # Train classifier
    archtype= platform.architecture()[0] # (’64bit’, ‘Windows7’)
  
    clsfile= '%s/classifier/classifier-%s.pcl'% (datadir,archtype)
    print('\n>> Training classifier')
    X_data_merged= np.concatenate( X_data )
    Y_data_merged= np.concatenate( Y_data ) 
    timer= qc.Timer()
    cls.fit( X_data_merged, Y_data_merged)
    print('Trained %d samples x %d dimension in %.1f sec'% \
        (X_data_merged.shape[0], X_data_merged.shape[1], timer.sec()))
    # set n_jobs = 1 for testing
    cls.n_jobs= 1
  
 
    classes= { c:cfg.tdef.by_value[c] for c in np.unique(Y_data) }
    #save FEATURES'PSD':
    data= dict( cls=cls, psde=psde, sfreq=sfreq, picks=picks, classes=classes,
        epochs=cfg.EPOCH, w_frames=w_frames, w_seconds=psdparams['wlen'],
        wstep=psdparams['wstep'], spfilter=spfilter, spchannels=spchannels, refchannel=None,
        tpfilter=tpfilter, notch=cfg.NOTCH_FILTER, triggers=cfg.tdef )  
    qc.make_dirs('%s/classifier'% datadir)
    qc.save_obj(clsfile, data)
  
    # Show top distinctive features
    if cfg.CLASSIFIER=='RF':
        print('\n>> Good features ordered by importance')
        keys, _= qc.sort_by_value( list(cls.feature_importances_), rev=True )
        if cfg.EXPORT_GOOD_FEATURES:
            gfout= open('%s/good_features.txt'% datadir, 'w')
  
        # reverse-lookup frequency from fft
        if type(wlen) is not list:
            fq= 0
            fq_res= 1.0 / psdparams['wlen']
            fqlist= []
            while fq <= psdparams['fmax']:
                if fq >= psdparams['fmin']: fqlist.append(fq)
                fq += fq_res
  
            for k in keys[:cfg.FEAT_TOPN]:
                ch,hz= qc.feature2chz(k, fqlist, picks, ch_names=raw.ch_names)
                print('%s, %.1f Hz  (feature %d)'% (ch,hz,k) )
                if cfg.EXPORT_GOOD_FEATURES:
                    gfout.write( '%s\t%.1f\n'% (ch, hz) )
              
            if cfg.EXPORT_GOOD_FEATURES:
                if cfg.CV_PERFORM is not None:
                    gfout.write('\nCross-validation performance: mean %.2f, std %.2f\n'%(cv_mean, cv_std) )
                gfout.close()
            print()
        else:
            print('Ignoring good features because of multiple epochs.')
 
    
    # Test file
    if len(cfg.ftest) > 0:
        raw_test, events_test= pu.load_raw('%s'%(cfg.ftest), spfilter)
 
        '''
        TODO: implement multi-segment epochs
        '''
        if type(cfg.EPOCH[0]) is list:
            print('MULTI-SEGMENT EPOCH IS NOT SUPPORTED YET.')
            sys.exit(-1)
 
        epochs_test= Epochs(raw_test, events_test, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1],\
            proj=False, picks=picks, baseline=None, preload=True, add_eeg_ref=False)
 
        
        psdfile= 'psd-test.pcl'
        if not os.path.exists(psdfile):
            print('\n>> Computing PSD for test set')
            X_test, y_test= pu.get_psd(epochs_test, psde, w_frames, int(sfreq/8))
            qc.save_obj(psdfile, {'X':X_test, 'y':y_test})
        else:
            print('\n>> Loading %s'% psdfile)
            data= qc.load_obj(psdfile)
            X_test, y_test= data['X'], data['y']
        
 
        score_test= cls.score( np.concatenate(X_test), np.concatenate(y_test) )
        print('Testing score', score_test)
 
        # running performance
        print('\nRunning performance over time')
        scores_windows= []
        timer= qc.Timer()
        for ep in range( y_test.shape[0] ):
            scores= []
            frames= X_test[ep].shape[0]
            timer.reset()
            for t in range(frames):
                X= X_test[ep][t,:]
                y= [y_test[ep][t]]
                scores.append( cls.score(X, y) )
                #print('%d /%d   %.1f msec'% (t,X_test[ep].shape[0],1000*timer.sec()) )
            print('Tested epoch %d, %.3f msec per window'%(ep, timer.sec()*1000.0/frames) )
            scores_windows.append(scores)
        scores_windows= np.array(scores_windows)  
        #print(scores_windows)  #predict result
 
#         ###############################################################################
#         # Plot performance over time
#         ###############################################################################
#         #w_times= (w_start + w_frames / 2.) / sfreq + epochs.tmin
#         step= float(epochs_test.tmax - epochs_test.tmin) / scores_windows.shape[1]
#         w_times= np.arange( epochs_test.tmin, epochs_test.tmax, step )
#         print(scores_windows)
#         print(w_times)
#         print(np.mean(scores_windows, 0))
#         plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
#         plt.axvline(0, linestyle='--', color='k', label='Onset')
#         plt.axhline(0.5, linestyle='-', color='k', label='Chance')
#         plt.xlabel('time (s)')
#         plt.ylabel('Classification accuracy')
#         plt.title('Classification score over time')
#         plt.legend(loc='lower right')
#         plt.show()


if __name__=='__main__':
    # load parameters
    import imp  
    cfg_module  = "config.py"
    if cfg_module[-3:]=='.py':
        cfg_module= cfg_module[:-3]
    cfg= imp.load_source(cfg_module, "./config.py")
    #print (qc.get_file_list(cfg.DATADIR, fullpath=True))
    # get train list
    ftrain= []
    for f in qc.get_file_list(cfg.DATADIR, fullpath=True):  
        print(f)
        f=f[0] #f if list getf[0]is filename
        print(f)
        if f[-4:] in ['.fif','.fiff','.pcl','.bdf','.gdf']:
            ftrain.append(f)
    #print (ftrain)


    # single run
    if True:
        print('ftrain')
        print(ftrain)
        run_trainer(cfg, ftrain, interactive=True)
        
        sys.exit()
