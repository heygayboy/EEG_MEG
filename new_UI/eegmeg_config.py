# -*- coding: utf-8 -*-
'''
Created on 2017年4月13日

@author: Administrator
'''

TRIGGER_DEVICE= 'USB2LPT'


# timings
refresh_rate= 0.05 # in seconds; min=0.01

t_init= 1 # time showing: 'Waiting to start', 15 s
t_gap= 2 # time showing: '1/20 trials'
t_cue= 2 # no bar, only red dot
t_dir_ready= 0 # green bar, no move
t_raise = 2
t_keep = 2

num_trials= 60
screen_width= 500
screen_height= 500