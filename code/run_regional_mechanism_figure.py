#!/usr/bin/env python3
"""Render descriptive reference-label window and fixed-geometry SI controls."""
from pathlib import Path
from io import BytesIO
import json
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':12,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})


def main():
    bs=json.loads((ROOT/'data/boundary_scaling.json').read_text())
    pol=json.loads((ROOT/'data/polarized.json').read_text())
    dr=json.loads((ROOT/'data/driven.json').read_text())
    figA=plt.figure(figsize=(6.0,4.6)); figBC,axbc=plt.subplots(1,2,figsize=(11.0,4.8))
    ax=[figA.add_subplot(111),axbc[0],axbc[1]]
    sty={'label_box':('o','#1f4e79','label box'),
         'label_disk':('s','#c0392b','label disk'),
         'label_rotated_box':('^','#e67e22','rotated label box')}
    for name,(marker,color,label) in sty.items():
        rows=[x for x in bs if x['name']==name]
        ax[0].loglog([x['nb'] for x in rows],[x['rms'] for x in rows],marker+'-',
                     color=color,ms=5,label=label,lw=1.4)
    ax[0].set_xlabel('boundary cells $n_b$')
    ax[0].set_ylabel(r'RMS $|\sum_\alpha W_\alpha|$')
    ax[0].legend(fontsize=10,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
    e=[x['eps'] for x in dr]; beta=[x['beta'] for x in pol]
    ax[1].semilogy(e,[x['regional']/dr[0]['regional'] for x in dr],'o-',color='#1f4e79',lw=1.4,
                   label='rebuilt Voronoi')
    ax[1].semilogy(beta,[x['reg']/pol[0]['reg'] for x in pol],'s-',color='#c0392b',lw=1.4,
                   label='fixed-geometry modulation')
    ax[1].axhline(1,color='k',ls=':',lw=.9)
    ax[1].set_xlabel(r'amplitude, $\epsilon$ or $\beta$')
    ax[1].set_ylabel('regional RMS gap /\nbaseline RMS gap')
    ax[1].legend(fontsize=9.5,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16))
    for values,rows,color,marker,label in [(e,dr,'#1f4e79','o','rebuilt'),(beta,pol,'#c0392b','s','modulated')]:
        ax[2].plot(values,[x['corr'] for x in rows],marker+'-',color=color,lw=1.4,label=label+r' $Q_{nn}$')
        ax[2].plot(values,[x['corr_centered'] for x in rows],marker+'--',color=color,lw=1.1,ms=4,
                   mfc='white',label=label+r' $Q^c_{nn}$')
    ax[2].axhline(0,color='k',lw=.9); ax[2].set_ylim(-.6,1.1)
    ax[2].set_xlabel(r'amplitude, $\epsilon$ or $\beta$')
    ax[2].set_ylabel('orientation statistic')
    ax[2].legend(fontsize=9.5,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
    for a,letter in zip(ax[1:],'AB'):
        a.text(-.14,1.10,letter,transform=a.transAxes,fontsize=17,fontweight='bold',va='top',ha='left')
    for a in ax:
        a.grid(False)
        for side in ['top','right']: a.spines[side].set_visible(False)
    figA.tight_layout();figBC.tight_layout()
    (ROOT/'figures').mkdir(exist_ok=True)
    for name,figure in [('figure_regional_mechanism_A',figA),('figure_regional_mechanism_BC',figBC)]:
        for ext in ['pdf','png']:
            stream=BytesIO();figure.savefig(stream,format=ext,dpi=200)
            with tempfile.NamedTemporaryFile(dir=ROOT/'figures',delete=False) as f:
                f.write(stream.getvalue());temporary=f.name
            os.replace(temporary,ROOT/f'figures/{name}.{ext}')
    plt.close('all')
    print('Rebuilt descriptive window plot and fixed-geometry SI controls; original scientific curves preserved.')


if __name__=='__main__':
    main()
