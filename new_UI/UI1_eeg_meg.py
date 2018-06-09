# -*- coding: utf-8 -*-
'''
Created on 2017年4月13日

@author: Administrator
'''
from __future__ import print_function
from __future__ import division

import cv2    #在Python中调用opencv
import cv2.cv as cv
import numpy as np
import sys, os, math, random, time, datetime
import q_common as qc
from socket import *

HOST = ''
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST,PORT)


tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

#initial
import imp   
cfg_module  = "train_config.py"
if cfg_module[-3:]=='.py':
    cfg_module= cfg_module[:-3]
cfg= imp.load_source(cfg_module, "./train_config.py")

keys= {'left':81,'right':83,'up':82,'down':84,'pgup':85,'pgdn':86,'home':80,'end':87,'space':32,'esc':27\
    ,',':44,'.':46,'s':115,'c':99,'[':91,']':93,'1':49,'!':33,'2':50,'@':64,'3':51,'#':35}
#create a sequence
dir_sequence= []
for x in range( cfg.trials_each ):
    dir_sequence.extend( cfg.directions )  #向数组dir_sequence= []加入0 1两个方向
random.shuffle( dir_sequence )   #shuffle() 方法将序列的所有元素随机排序
event= 'start'
trial= 1
#image
img= np.zeros((cfg.screen_height,cfg.screen_width,3), np.uint8)  
color= dict(G=(20,140,0), B=(210,0,0), R=(0,50,200), Y=(0,215,235), K=(0,0,0), W=(255,255,255), w=(200,200,200))

print('waiting for connection...')
tcpCliSock,addr = tcpSerSock.accept()
print('...connected from: ',addr)

def screen_erase(img):
    cv2.rectangle( img, (0,0), (500,500), (0,0,0), -1 )

def draw_cue(img, box=color['R'], cross=color['W']):
    cv2.rectangle( img, (100,200), (400,300), color['w'], -1 )
    cv2.rectangle( img, (200,100), (300,400), color['w'], -1 )
    cv2.rectangle( img, (200,200), (300,300), box, -1 )
    #cv2.circle( img, (250,250), 10, color['Y'], -1 )
    cv2.rectangle( img, (240,248), (260,252), cross, -1 )
    cv2.rectangle( img, (248,240), (252,260), cross, -1 )
    
cv2.namedWindow("mi")  
cv2.moveWindow("mi", 430, 130)  #Moves window to the specified position  


timer_refresh= qc.Timer()  #class Timer:
timer_trigger= qc.Timer()
timer_dir= qc.Timer()

 
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

    elif event=='gap_s':
        cv2.putText(img, 'Trial %d / %d'%(trial,cfg.num_trials), (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        event= 'gap'
    elif event=='gap' and timer_trigger.sec() > cfg.t_gap:
        event= 'cue'
        screen_erase(img)
        draw_cue(img)
        timer_trigger.reset()
    elif event=='cue' and timer_trigger.sec() > cfg.t_cue:
        event= 'dir_r'
        dir= dir_sequence[trial-1]
        if dir==0: # left
            cv2.rectangle( img, (100,200), (200,300), color['B'], -1)
        elif dir==1: # right
            cv2.rectangle( img, (300,200), (400,300), color['B'], -1)
        elif dir==2: # up
            cv2.rectangle( img, (200,100), (300,200), color['B'], -1)
        elif dir==3: # down
            cv2.rectangle( img, (200,300), (300,400), color['B'], -1)
        timer_trigger.reset()
    elif event=='dir_r' and timer_trigger.sec() > cfg.t_dir_ready:
        screen_erase(img)
        draw_cue(img, box=(0,170,0) )
        event= 'dir'
        timer_trigger.reset()
        timer_dir.reset()
        tcpCliSock.send("Begin to record")
        '''
        if dir==0: # left
            trigger.signal(tdef.LEFT_GO)
        elif dir==1: # right
            trigger.signal(tdef.RIGHT_GO)
        elif dir==2: # up
            trigger.signal(tdef.UP_GO)
        elif dir==3: # down
            trigger.signal(tdef.DOWN_GO)
        '''
    elif event=='dir' and timer_trigger.sec() > cfg.t_dir:
        event= 'gap_s'
        screen_erase(img)
        trial += 1
        tcpCliSock.send("end to record")
        timer_trigger.reset()

    # protocol
    if event=='dir':
        dx= min( 100, int( 100.0 * timer_dir.sec() / cfg.t_dir ) + 1 )
        if dir==0: # L
            cv2.rectangle( img, (200-dx,200), (200,300), color['B'], -1 )
        if dir==1: # R
            cv2.rectangle( img, (300,200), (300+dx,300), color['B'], -1 )
        if dir==2: # U
            cv2.rectangle( img, (200,200-dx), (300,200), color['B'], -1 )
        if dir==3: # D
            cv2.rectangle( img, (200,300), (300,300+dx), color['B'], -1 )

    if event=='start':
        cv2.putText(img, 'Waiting to start', (120,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("mi", img)
    key= 0xFF & cv2.waitKey(1)

    if key==keys['esc']:
        break

cv2.destroyWindow("mi")
tcpSerSock.close()
