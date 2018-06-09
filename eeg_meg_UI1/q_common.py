# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import sys, os, time, math, inspect, itertools
import traceback
import numpy as np
try:
    import cPickle as pickle # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle # Python 3 (C version is the default)
    
def get_file_list( path, fullpath=True):  #DATADIR= r'D:\EEG_Data\qs9\session2\fif'
    """
    Get files with or without full path.
    """
    filelist=[]
    path= path.replace('\\','/')   
    if not path[-1]=='/': path += '/'

    if fullpath==True:
        for f in os.listdir(path):
            if os.path.isfile(path+'/'+f) and f[0]!='.':
                filelist.append([path+f])  
    else:
        filelist= [f for f in os.listdir(path) if os.path.isfile(path+'/'+f) and f[0]!='.']

    return sorted( filelist )

def load_obj(fname):
    """
    Read python object from a file
    """
    try:
        with open(fname, 'rb') as fin:
            return pickle.load(fin)
    except:
        msg= 'load_obj(): Cannot load pickled object file "%s". The error was:\n%s\n%s'% \
            (fname,sys.exc_info()[0],sys.exc_info()[1])
        print_error(msg)
        sys.exit(-1)
        
def print_error(msg):
    """
    Print message with the caller's name
    """
    import inspect
    callerName= inspect.stack()[1][3]
    print('\n>> Error in "%s()":\n%s\n'% (callerName,msg) )
    
def make_dirs(dirname, delete=False):
    """
    Recusively create directories.
    if delete=true, directory will be deleted first if exists.
    """
    import shutil
    if os.path.exists(dirname) and delete==True:
        try:
            shutil.rmtree(dirname)
        except OSError:
            print_error('Directory was not completely removed. (Perhaps a Dropbox folder?). Continuing.')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def sort_by_value(s, rev=False):
    """
    Sort dictionary or list by value and return a sorted list of keys and values.
    Values must be hashable and unique.
    """

    assert type(s)==dict or type(s)==list, 'Input must be a dictionary or list.'
    if type(s)==list:
        s= dict(enumerate(s))
    #if Q_VERBOSE > 1 and not len(set(s.values()))==len(s.values()):
    #    print('>> Warning: %d overlapping values. Length will be shorter.'% \
    #        (len(s.values())-len(set(s.values()))+1))

    s_rev= dict((v,k) for k,v in s.items())
#     if Q_VERBOSE > 0 and not len(s_rev)==len(s):
#         print('** WARNING sort_by_value(): %d identical values'% (len(s.values())-len(set(s.values()))+1) )
    values= sorted(s_rev, reverse=rev)
    keys= [s_rev[x] for x in values]
    return keys, values

def feature2chz(x, fqlist, picks, ch_names=None):
    """
    Label channel, frequency pair for PSD feature indices

    Input
    ------
    x: feature index
    picks: channels used (channel 0 being trigger channel)
    fqlist: list of frequency bands
    ch_names: list of complete channel names

    Output
    -------
    (channel, frequency)

    """

    n_fq= len(fqlist)
    hz= fqlist[ x % n_fq ]
    ch= int( x / n_fq ) # 0-based indexing
    ch_names= np.array(ch_names)
    try:
        if ch_names is not None:
            return ch_names[picks[ch]], hz
        else:
            return picks[ch], hz
    except:
        traceback.print_exc()
        raise RuntimeError, '\n**** Error in feature2chz(). ****'

def save_obj(fname, obj, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save python object into a file
    Set protocol=2 for Python 2 compatibility
    """
    with open(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

class Timer:
    """
    Timer class

    if autoreset=True, timer is reset after any member function call

    """

    def __init__(self, autoreset=False):
        self.autoreset= autoreset
        self.reset()
    def sec(self):
        read= time.time() - self.ref
        if self.autoreset: self.reset()
        return read
    def msec(self):
        return self.sec()*1000.0
    def reset(self):
        self.ref= time.time()
    def sleep_atleast(self, sec):
        """
        Sleep up to sec seconds
        It's more convenient if autoreset=True
        """
        timer_sec= self.sec()
        if timer_sec < sec:
            time.sleep( sec - timer_sec )
            if self.autoreset: self.reset()

# print_c: print texts in color
try:
    import colorama
    colorama.init()
    def print_c(msg, color, end='\n'):
        """
        Colored print using colorama.

        Fullset:
            https://pypi.python.org/pypi/colorama
            Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            Style: DIM, NORMAL, BRIGHT, RESET_ALL

        TODO:
            Make it using *args and **kwargs to support fully featured print().

        """
        if type(msg) not in [str, unicode]:
            raise RuntimeError, 'msg parameter must be a string. Recevied type %s'% type(msg)
        if type(color) not in [str, unicode] and len(color) != 1:
            raise RuntimeError, 'color parameter must be a single color code. Received type %s'% type(color)

        if color.upper()=='B':
            c= colorama.Fore.BLUE
        elif color.upper()=='R':
            c= colorama.Fore.RED
        elif color.upper()=='G':
            c= colorama.Fore.GREEN
        elif color.upper()=='Y':
            c= colorama.Fore.YELLOW
        elif color.upper()=='W':
            c= colorama.Fore.WHITE
        elif color.upper()=='C':
            c= colorama.Fore.CYAN
        else:
            assert False, 'pu.print_color(): Wrong color code.'
        print( colorama.Style.BRIGHT + c + msg + colorama.Style.RESET_ALL, end=end )
except ImportError:
    print('\n\n*** WARNING: colorama module not found. print_c() will ignore color codes. ***\n')
    def print_c(msg, color, end='\n'):
        print( msg, end=end )