# -*- coding: utf-8 -*-
'''
Created on 2017年4月13日

@author: Administrator
'''

TRIGGER_DEVICE= 'USB2LPT'


# classes
directions= [0,1] # 0:L, 1:R, 2:U, 3:D

trials_each= 30 # number of trials for each action

# timings
refresh_rate= 0.05 # in seconds; min=0.01

t_init= 5 # time showing: 'Waiting to start', 15 s
t_gap= 2 # time showing: '1/20 trials'
t_cue= 2 # no bar, only red dot
t_dir_ready= 2 # green bar, no move
t_dir= 10 # blue bar

num_directions= len(directions)
num_trials= len(directions) * trials_each
screen_width= 500
screen_height= 500