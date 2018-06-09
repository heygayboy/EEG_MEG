# -*- coding: utf-8 -*-
'''
Created on 2017��4��13��

@author: Administrator
'''
from __future__ import print_function
from __future__ import division

import cv2    #��Python�е���opencv
import cv2.cv as cv
import numpy as np
import sys, os, math, random, time, datetime
import q_common as qc
from socket import *
import pycnbi_config
import pyLptControl
from triggerdef_16 import TriggerDef
import mne.io, mne.viz
import q_common as qc
import bgi_client

TRIGGER_DEVICE= 'USB2LPT'

#initial
import imp   
cfg_module  = "eegmeg_config.py"
if cfg_module[-3:]=='.py':
    cfg_module= cfg_module[:-3]
cfg= imp.load_source(cfg_module, "./eegmeg_config.py")

print(cfg.t_keep)
keys= {'left':81,'right':83,'up':82,'down':84,'pgup':85,'pgdn':86,'home':80,'end':87,'space':32,'esc':27\
    ,',':44,'.':46,'s':115,'c':99,'[':91,']':93,'1':49,'!':33,'2':50,'@':64,'3':51,'#':35}
event= 'start'
trial= 1
#image
img= np.zeros((cfg.screen_height,cfg.screen_width,3), np.uint8)  
color= dict(G=(20,140,0), B=(210,0,0), R=(0,50,200), Y=(0,215,235), K=(0,0,0), W=(255,255,255), w=(200,200,200))
run_serve = True

def run_serve():
    HOST = ''
    PORT = 21567
    BUFSIZ = 1024
    ADDR = (HOST,PORT)
     
     
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)
    print('waiting for connection...')
    tcpCliSock,addr = tcpSerSock.accept()
    print('...connected from: ',addr)
    return tcpCliSock

if run_serve:
    tcpCliSock = run_serve()
    
def screen_erase(img):
    cv2.rectangle( img, (0,0), (500,500), (0,0,0), -1 )

def draw_cue(img, box=color['R'], cross=color['W']):
    cv2.rectangle( img, (100,200), (400,300), color['w'], -1 )
    cv2.rectangle( img, (250,200), (400,300), color['Y'], -1 )

trigger= pyLptControl.Trigger('USB2LPT', 0x378)
if trigger.init(50)==False:
    print('\n# Error connecting to USB2LPT device. Use a mock trigger instead?')
    raw_input('Press Ctrl+C to stop or Enter to continue.')
    trigger= pyLptControl.MockTrigger()
    trigger.init(50)
        
cv2.namedWindow("eeg_meg")  
cv2.moveWindow("eeg_meg", 430, 130)  #Moves window to the specified position  


timer_refresh= qc.Timer()  #class Timer:
timer_trigger= qc.Timer()
timer_dir= qc.Timer()
tdef= TriggerDef()

# start
while trial <= cfg.num_trials: 
    timer_refresh.sleep_atleast(0.2)
    #print(timer_refresh.sec())
    timer_refresh.reset()

    # segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
    if event=='start' and timer_trigger.sec() > cfg.t_init:
        event= 'gap_s'
        screen_erase(img)
        timer_trigger.reset()
        trigger.signal(tdef.INIT)

    elif event=='gap_s':
        cv2.putText(img, 'Trial %d / %d'%(trial,cfg.num_trials), (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        event= 'gap'
    elif event=='gap' and timer_trigger.sec() > cfg.t_gap:
        event= 'cue'
        screen_erase(img)
        draw_cue(img)
        trigger.signal(tdef.CUE)
        cv2.putText(img, 'raise', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, 'keep', (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        timer_trigger.reset()
    elif event=='cue' and timer_trigger.sec() > cfg.t_cue:
        event= 'dir_r'
        draw_cue(img)
        trigger.signal(tdef.raise_ready)
        timer_trigger.reset()
    elif event=='dir_r' and timer_trigger.sec() > cfg.t_dir_ready:
        screen_erase(img)
        draw_cue(img, box=(0,170,0) )
        
        event= 'raise'
        timer_trigger.reset()
        timer_dir.reset()
        tcpCliSock.send("begin to record")
        trigger.signal(tdef.raise_begin) 

    elif event == 'raise' and timer_trigger.sec() > cfg.t_raise:
        event = 'keep'
        timer_trigger.reset()
        timer_dir.reset()
        
    elif event=='keep' and timer_trigger.sec() > cfg.t_keep:
        event= 'gap_s'
        screen_erase(img)
        trial += 1
        trigger.signal(tdef.BLANK)
        tcpCliSock.send("end to record")
        timer_trigger.reset()

    # protocol
    if event == 'raise':
        dx1= min( 150 , int( 150.0 * timer_dir.sec() / cfg.t_raise ) + 1 )
        cv2.rectangle( img, (100,200), (100+dx1,300), color['B'], -1 )
        cv2.putText(img, 'raise', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, 'keep', (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    if event=='keep':
        dx2= min( 150, int( 150.0 * timer_dir.sec() / cfg.t_keep ) + 1 )
        cv2.rectangle( img, (250,200), (250+dx2,300), color['G'], -1 )
        cv2.putText(img, 'raise', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, 'keep', (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if event=='start':
        cv2.putText(img, 'Waiting to start', (120,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("eeg_meg", img)
    key= 0xFF & cv2.waitKey(1)

    if key==keys['esc']:
        break

cv2.destroyWindow("eeg_meg")
tcpCliSock.close()
