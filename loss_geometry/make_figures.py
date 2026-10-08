"""Render loss-geometry figures from saved numerical data."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from geometry import angular_occupancy, exact_q_closed

ROOT = Path(__file__).parent
DATA, FIG = ROOT/'results', ROOT/'figures'
FIG.mkdir(exist_ok=True)
WIDTH = 7.05
BASE, TICK, PANEL = 7.6, 7.0, 8.4
INK, PETROL, CLAY, PLUM, SLATE = '#1b1b1b', '#355f6b', '#b77a65', '#7d708e', '#929aa0'
LOSS_CMAP = LinearSegmentedColormap.from_list('loss_sage', [
    (0.00, '#ffffff'), (0.06, '#f0f3ed'), (0.25, '#c9dcd4'),
    (0.55, '#8ab0a6'), (0.80, '#5f8886'), (1.00, '#355f6b')])
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{bm}',
    'font.size': BASE, 'axes.labelsize': BASE, 'axes.linewidth': 0.5,
    'axes.labelpad': 2.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'lines.linewidth': 0.7, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 1.8, 'ytick.major.size': 1.8,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.minor.size': 1.1, 'ytick.minor.size': 1.1,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
    'legend.fontsize': BASE, 'legend.frameon': False,
    'legend.borderaxespad': 0.5, 'legend.handlelength': 1.7,
    'legend.handletextpad': 0.5, 'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.3, 'savefig.dpi': 430,
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def save(fig, name):
    # Fixed page dimensions preserve physical text size on manuscript insertion.
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'), dpi=430)
    plt.close(fig)


def panel(fig, letter, x, y):
    fig.text(x, y, f'({letter})', ha='left', va='top', fontsize=PANEL)


def field_data(name):
    return np.load(DATA/(name+'.npz'))


def show_field(ax, data, limit, show_y=True):
    x, field = data['x'], data['field']
    h=x[1]-x[0]
    im=ax.imshow(field, origin='lower', extent=[x[0]-h/2,x[-1]+h/2]*2,
        cmap=LOSS_CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax.set(xlim=(-limit,limit), ylim=(-limit,limit), xlabel=r'$x/L$',
           xticks=[-.4,0,.4], yticks=[-.4,0,.4], aspect='equal')
    if show_y:
        ax.set_ylabel(r'$y/L$')
    else:
        ax.tick_params(labelleft=False)
    return im


def map_row(fig, names, limit, height):
    lefts=[.085,.36,.635]
    width=.215
    axes=[]
    for i,(name,left) in enumerate(zip(names,lefts)):
        ax=fig.add_axes([left,.55,width,width*WIDTH/height])
        im=show_field(ax,field_data(name),limit,show_y=(i==0))
        panel(fig,chr(97+i),left-.058,.986)
        axes.append(ax)
    cax=fig.add_axes([.89,.585,.012,.29])
    cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1])
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(width=.4,length=1.6,pad=2,labelsize=TICK)
    return axes,cb


def exact_design():
    profiles=json.loads((DATA/'annular_profiles_w005.json').read_text())
    rings=pd.read_csv(DATA/'annular_realization.csv')
    fig=plt.figure(figsize=(WIDTH,2.24))
    lefts=[.075,.405,.735]
    axes=[fig.add_axes([left,.235,.23,.66]) for left in lefts]
    for i,left in enumerate(lefts):
        panel(fig,chr(97+i),left-.055,.987)
    delta=np.geomspace(.001,1,400)
    axes[0].semilogx(delta,[exact_q_closed(d) for d in delta],color=PETROL,lw=.95,
                    label=r'$C_\star$')
    eight=rings.query('rings == 8').sort_values('w')
    axes[0].semilogx(eight.w,eight.q,'o',color=CLAY,ms=3.0,mew=0,label='8 annuli')
    axes[0].set(xlabel=r'$\delta=w/L$',ylabel=r'$\mathcal{Q}$',ylim=(0,6.1),
                yticks=[0,2,4,6])
    axes[0].legend(loc='lower left')
    r=np.geomspace(.002,.501,1000)
    axes[1].semilogx(r,angular_occupancy(r,.05),color=PETROL,lw=.95,label='Exact')
    edges=np.array(profiles['8']['edges'])
    levels=np.array(profiles['8']['levels'])
    axes[1].stairs(levels,edges,color=CLAY,lw=.85,label='8 annuli',baseline=None)
    axes[1].set(xlim=(.004,.55),ylim=(-.025,1.06),xlabel=r'$r/L$',
                ylabel=r'$\gamma/\gamma_{\max}$',yticks=[0,.5,1])
    axes[1].legend(loc='upper right')
    row=rings.query('w == 0.05')
    axes[2].plot(row.rings,100*(1-row.retention),'o-',color=PETROL,lw=.85,ms=3,mew=0)
    axes[2].set_xscale('log',base=2)
    axes[2].set_xticks([2,4,8,16,32],[2,4,8,16,32])
    axes[2].set(xlabel='Number of annuli',ylabel=r'$1-\mathcal{Q}/C_\star$ (\%)',
                ylim=(0,20),yticks=[0,5,10,15,20])
    save(fig,'01_exact_limit_and_design')


def fixed_statistics():
    names=['radial_optimum','fixed_support_radial_reverse','fixed_support_shuffle']
    labels=['Radial','Reversed','Shuffled']
    colors=[PETROL,PLUM,CLAY]
    fig=plt.figure(figsize=(WIDTH,3.90))
    _,cb=map_row(fig,names,limit=.56,height=3.90)
    cb.set_label(r'$\gamma/\gamma_{\max}$',fontsize=BASE,labelpad=4)
    ax=fig.add_axes([.085,.135,.765,.285])
    panel(fig,'d',.027,.46)
    for name,label,color in zip(names,labels,colors):
        data=field_data(name)
        h=data['x'][1]-data['x'][0]
        B=h*np.linalg.norm(data['field'])
        ax.plot(np.degrees(data['theta']),data['response']/B,
                color=color,lw=.95,label=label)
    ax.set(xlim=(0,180),ylim=(0,1.95),xlabel=r'$\theta$ (deg)',
        ylabel=r'$v_F\mathcal{D}(\theta)/\|\gamma\|_2$',
        xticks=[0,45,90,135,180],yticks=[0,.5,1,1.5])
    ax.legend(loc='lower center',ncols=3,bbox_to_anchor=(.5,.015))
    save(fig,'02_fixed_statistics_response')


def binary_geometry():
    names=['binary_disk','binary_deltoid','binary_deltoid_optimized']
    labels=['Disk','Deltoid','Optimized']
    colors=[SLATE,CLAY,PETROL]
    fig=plt.figure(figsize=(WIDTH,3.90))
    _,cb=map_row(fig,names,limit=.60,height=3.90)
    cb.set_label(r'$\gamma/\gamma_0$',fontsize=BASE,labelpad=4)
    ax=fig.add_axes([.085,.135,.475,.285])
    covax=fig.add_axes([.705,.135,.245,.285])
    panel(fig,'d',.027,.46)
    panel(fig,'e',.64,.46)
    threshold=np.linspace(0,.56,600)
    for name,label,color in zip(names,labels,colors):
        data=field_data(name)
        ax.plot(np.degrees(data['theta']),data['response'],color=color,lw=.95,label=label)
        coverage=np.mean(data['response'][:,None]>=threshold[None,:],axis=0)
        covax.step(threshold,coverage,where='post',color=color,lw=.95)
    ax.axhline(.4,color=PLUM,ls=(0,(3,2)),lw=.6,zorder=0)
    ax.set(xlim=(0,180),ylim=(0,.55),xlabel=r'$\theta$ (deg)',
        ylabel=r'$v_F\mathcal{D}/(\gamma_0 L)$',xticks=[0,45,90,135,180],
        yticks=[0,.2,.4])
    ax.legend(loc='lower center',ncols=3,bbox_to_anchor=(.5,.035),columnspacing=1.0)
    covax.axvline(.4,color=PLUM,ls=(0,(3,2)),lw=.6,zorder=0)
    covax.set(xlabel=r'$\beta v_F/(\gamma_0 L)$',ylabel='Angular coverage',
        ylim=(0,1.055),xlim=(0,.56),xticks=[0,.2,.4],yticks=[0,.5,1])
    save(fig,'03_binary_geometry_and_coverage')


def peak_constraint():
    cap=pd.read_csv(DATA/'rate_cap_radial_family.csv')
    fig=plt.figure(figsize=(WIDTH,2.34))
    ax=fig.add_axes([.085,.23,.355,.66])
    peakax=fig.add_axes([.59,.23,.36,.66])
    panel(fig,'a',.027,.987)
    panel(fig,'b',.52,.987)
    ax.semilogx(cap.w,cap.exact_q,color=PETROL,lw=.95,label='Exact optimum')
    ax.semilogx(cap.w,cap.q,color=CLAY,lw=.95,ls=(0,(4,2)),label='Capped radial')
    ax.set(xlabel=r'$w/L$',ylabel=r'$\mathcal{Q}$',ylim=(0,5.6),yticks=[0,1,2,3,4,5])
    ax.legend(loc='lower left')
    peak=np.sqrt(2*np.pi)*.08/(cap.w*cap.exact_q)
    peakax.loglog(cap.w,peak,color=PETROL,lw=.95,label='Required peak')
    peakax.axhline(1,color=CLAY,lw=.8,ls=(0,(4,2)),label='Available peak')
    peakax.set(xlabel=r'$w/L$',ylabel=r'$\gamma_{\max} L/v_F$')
    peakax.legend(loc='upper right')
    save(fig,'04_peak_rate_constraint')


if __name__ == '__main__':
    exact_design()
    fixed_statistics()
    binary_geometry()
    peak_constraint()
