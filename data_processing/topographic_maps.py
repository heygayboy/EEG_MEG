# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division


import numpy as np
import mne
import matplotlib.pyplot as plt
import pycnbi_utils as pu
from mne import Epochs, pick_types
import q_common as qc
from mne.viz import topomap
from mne.stats import spatio_temporal_cluster_test
from mne.channels import read_ch_connectivity
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable

connectivity, ch_names = read_ch_connectivity('biosemi16')



# load parameters
import imp  
cfg_module  = "config.py"
if cfg_module[-3:]=='.py':
    cfg_module= cfg_module[:-3]
cfg= imp.load_source(cfg_module, "./config.py")

spfilter= cfg.SP_FILTER
tpfilter= cfg.TP_FILTER
triggers= { cfg.tdef.by_value[c]:c for c in set(cfg.TRIGGER_DEF) }
print(triggers)

# ftrain =r'D:\data\Records\fif\20170309-195357-raw.fif'
ftrain= []
for f in qc.get_file_list(cfg.DATADIR, fullpath=True):  
    f=f[0] #f if list getf[0]is filename
    if f[-4:] in ['.fif','.fiff','.pcl','.bdf','.gdf']:
        ftrain.append(f)

# Load multiple files
multiplier= 1
raw, events= pu.load_multi(ftrain, spfilter=spfilter, multiplier=multiplier)
event_id = triggers
picks= pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads') 
print(picks)
epochs_train= Epochs(raw, events, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1], proj=False,\
            picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False, detrend=None)
# print(epochs_train)
# <Epochs  |  n_events : 300 (all good), tmin : 1.0 (s), tmax : 2.0 (s), baseline : None, ~18.8 MB, data loaded,
#  'LEFT_GO': 150, 'RIGHT_GO': 150>

epochs_train.equalize_event_counts(event_id)
print(epochs_train)
condition_names = 'LEFT_GO', 'RIGHT_GO'
X = [epochs_train[k].get_data() for k in condition_names]
X = [np.transpose(x, (0, 2, 1)) for x in X]  # transpose for clustering




# set cluster threshold
threshold = 50.0  # very high, but the test is quite sensitive on this data
# set family-wise p-value
p_accept = 0.001

cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=threshold, tail=1,
                                             n_jobs=1,
                                             connectivity=connectivity)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]






# configure variables for visualization
times = epochs_train.times * 1e3
colors = 'r', 'r', 'steelblue', 'steelblue'
linestyles = '-', '--', '-', '--'

# grand average as numpy arrray
grand_ave = np.array(X).mean(axis=1)

# get sensor positions via layout
print(epochs_train.info)
layout=mne.find_layout(epochs_train.info,'eeg')
print(layout)
pos = mne.find_layout(epochs_train.info,'eeg').pos

# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at significant sensors
    signals = grand_ave[..., ch_inds].mean(axis=-1)
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    title = 'Cluster #{0}'.format(i_clu + 1)
    fig.suptitle(title, fontsize=14)

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(f_map, pos, mask=mask, axes=ax_topo,
                            cmap='Reds', vmin=np.min, vmax=np.max)

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel('Averaged F-map ({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]
    ))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    for signal, name, col, ls in zip(signals, condition_names, colors,
                                     linestyles):
        ax_signals.plot(times, signal, color=col, linestyle=ls, label=name)

    # add information
    ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
    ax_signals.set_xlim([times[0], times[-1]])
    ax_signals.set_xlabel('time [ms]')
    ax_signals.set_ylabel('evoked magnetic fields [fT]')

    # plot significant time range
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
    ax_signals.legend(loc='lower right')
    ax_signals.set_ylim(ymin, ymax)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()



#topomap.plot_epochs_psd_topomap(epochs_train)

#epochs_train.plot_psd_topomap(bands=None, vmin, vmax, tmin, tmax, proj, bandwidth, adaptive, low_bias, normalization, ch_type, layout, cmap, agg_fun, dB, n_jobs, normalize, cbar_fmt, outlines, axes, show, verbose)



