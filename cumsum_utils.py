# Various graphing utilities

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
np.set_printoptions(edgeitems=40, linewidth=140, suppress=True)
pd.set_option('display.min_rows', 40)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: '%.8f' % x)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', 100)

import seaborn as sns
sns.set(rc={"figure.dpi": 150, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")


def where(cond):
    'Returns the indices where `cond` is true.'
    return np.nonzero(cond)[0]

def wherein(elements, test_elements):
    'Returns the index of each `elements` item in `test_elements`'
    return np.array([where(test_elements == num)[0] for num in elements])


def rowplot(plotfns, sharex=False, sharey=False, figsize=(12,4), share_aspect=False):
    num_plots = len(plotfns)
    yx_aspect = None

    fig, axs = plt.subplots(1, num_plots, figsize=figsize, sharex=sharex, sharey=sharey)
    
    base = 100 + 10*num_plots
    for i,plotfn in enumerate(plotfns):
        plt.subplot(base+i+1)
        if type(plotfn) is list:
            for fn in plotfn:
                fn()
        else:
            plotfn()

        left, right = plt.xlim()
        bottom, top = plt.ylim()
        if share_aspect and yx_aspect:
            curr_aspect = (top - bottom) / (right - left)
            if curr_aspect < yx_aspect:
                new_height = yx_aspect * (right - left)
                mid_y = (top + bottom) / 2
                plt.ylim(mid_y - new_height/2, mid_y + new_height/2)
            else:
                new_width = (top - bottom) / yx_aspect
                mid_x = (left + right) / 2
                plt.xlim(mid_x - new_width/2, mid_x + new_width/2)                
        else:
            yx_aspect = (top - bottom) / (right - left)  # Only compute on first plot
    
    return fig, axs


def listplot(ys, show=None, xscale=None, flip_axes=False, line=False, first=False, s=1, 
             xlabel='', ylabel='', title='', 
             xlim=None, ylim=None, means=False, h=None, v=None, **kargs):
    if type(ys[0]) is not np.ndarray and type(ys[0]) is not list:
        ys = [ys]
    
    for i,y in enumerate(ys):
        if line:
            width = 3 if first and i == 0 else 1
            if flip_axes:
                plt.plot(y, range(len(y)), marker='.', markersize=1, linewidth=width, **kargs)
            else:
                plt.plot(range(len(y)), y, marker='.', markersize=1, linewidth=width, **kargs)
        else:
            if flip_axes:
                plt.scatter(y, range(len(y)), s=s, marker='.', **kargs)
            else:
                plt.scatter(range(len(y)), y, s=s, marker='.', **kargs)
        
        if means:
            hline(np.mean(y))
        
        if show is not None:
            colors = 'rkymgc'
            for cols,c in zip(show, colors):
                plt.scatter(cols, y[cols], c=c, s=20, marker='.')

    if xscale: plt.xticks(plt.xticks()[0][1:-1], [int(xscale*x) for x in plt.xticks()[0][1:-1]])
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if title: plt.title(title)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if h is not None: hline(h)
    if v is not None: vline(v)

        
def scatter(x, y, s=1, show=None, xlabel='', ylabel='', title='', xlim=None, ylim=None, **kargs):
    ax = plt.scatter(x, y, s=s, marker='.', **kargs)

    colors = 'rkymgc'
    show = [] if show is None else show
    for s,c in zip(show, colors):
        scatter(x[s], y[s], c=c, s=20)

    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if title: plt.title(title)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    return ax


def hline(ys, c='g', alpha=0.5, **kargs):
    if type(ys) is not list and type(ys) is not np.ndarray: ys = [ys]
    for y in ys:
        plt.axhline(y, c=c, alpha=alpha, **kargs);

def vline(xs, c='g', alpha=0.5, **kargs):
    if type(xs) is not list and type(xs) is not np.ndarray: xs = [xs]
    for x in xs:
        plt.axvline(x, c=c, alpha=alpha, **kargs);

def iline(low, high, c='g', alpha=0.5, **kargs):
    plt.plot([low,high], [low,high], c=c, alpha=alpha, **kargs)
