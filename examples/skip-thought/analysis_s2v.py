#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# ----------------------------------------------------------------------------
"""
    $python analysis_s2v.py --s2v s2v.pkl \
                            --vector_2d_file vector_2D_sent.pkl
"""

import cPickle
import numpy as np
import os
from configargparse import ArgumentParser
from itertools import cycle
from textwrap import wrap
from sent_vectors import SentenceVector


class Annotation(object):
    def __init__(self, fig, ax, point_x, point_y, text):
        self.fig = fig
        self.ax = ax
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.point_x = point_x
        self.point_y = point_y
        self.text = text
        self.annot = None
        assert len(text) == len(point_x) == len(point_y)

    def find_nearest(self, target_x, target_y):
        idx = np.argmin(np.sqrt(np.square(self.point_x - target_x) +
                                np.square(self.point_y - target_y)))
        return idx

    def __call__(self, event):

        if self.annot is not None:
            try:
                self.annot.remove()
            except:
                pass
            finally:
                self.annot = None

        n_idx = self.find_nearest(event.xdata, event.ydata)
        sent_wrap = "\n".join(wrap(self.text[n_idx][0][0], 60))

        self.annot = plt.text(self.point_x[n_idx], self.point_y[n_idx], sent_wrap,
                              fontdict={'fontsize': 16, 'fontweight': 'bold'},
                              va='baseline', ha='center')
        plt.draw()

cycol = cycle('rbgcmykw').next

parser = ArgumentParser(__doc__)
parser.add_argument('--s2v', required=True,
                    help='the encoded sentence vectors saved from inference script')
parser.add_argument('--vector_2d_file', required=True,
                    help='the 2D vectors to save for analysis')
args = parser.parse_args()

sent_vec, sent_text = cPickle.load(open(args.s2v, 'rb'))

print "\nTotal sentences: {}".format(len(sent_text))

plot_only = 3000
# vis_range = range(plot_only)
# vis_range = np.random.choice(range(len(sent_text)), size=plot_only, replace=False)

s2v = SentenceVector(sent_vec, sent_text)
sim_idx = s2v.find_similar_with_idx(1000, n=plot_only)
vis_range = sim_idx

sent_vec = np.asarray(sent_vec).astype('float64')
low_dim_embs = None

sent_text_vis = sent_text[vis_range]
test_topic = []  # ['pricing', 'risk', 'growth']
sent_list = dict()
for wrd in test_topic:
    sent_list[wrd] = [idx for idx, sent in enumerate(sent_text_vis) if wrd in sent[0][0]]

try:
    from tsne import bh_sne
    import matplotlib.pyplot as plt
    low_dim_embs_all = bh_sne(sent_vec)

    cPickle.dump((low_dim_embs_all, sent_text), open(args.vector_2d_file, 'wb'))

    low_dim_embs = bh_sne(sent_vec[vis_range])
    low_dim_embs_x = low_dim_embs[:, 0]
    low_dim_embs_y = low_dim_embs[:, 1]

    fig = plt.figure(figsize=(15, 15))  # in inches
    ax = fig.add_subplot(111)
    ax.plot(low_dim_embs_x, low_dim_embs_y, 'bo', alpha=0.3)

    for wrd in test_topic:
        ax.plot(low_dim_embs_x[sent_list[wrd]], low_dim_embs_y[sent_list[wrd]], 'o',
                markersize=10, color=cycol(), alpha=0.5, label=wrd)

    plt.legend(loc='lower left', numpoints=1, ncol=8, fontsize=10, bbox_to_anchor=(0, 0))

    Annotation(fig, ax, low_dim_embs_x, low_dim_embs_y, sent_text_vis)

    plt.show()

except ImportError:
    print("Please install tsne and matplotlib to visualize embeddings.")
