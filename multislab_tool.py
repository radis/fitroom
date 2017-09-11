# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:41:43 2017

@author: erwan

Description
-----

a window to decompose the center slab along the different slabs

"""

import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor


class MultiSlabPlot():
    
    def __init__(self, 
                 plotquantity='radiance', unit= 'mW/cm2/sr/nm', 
                 normalizer=None,
                 wexp=None, Iexpcalib=None, wexp_shift=0,
                 nfig=None,
                 slit=None,  # add in a ExperimentConditions class?
                 slit_options={'norm_by':'area', 'shape':'triangular',
                               'slit_unit':'nm'},
                 ):
        
        # Init variables
        self.line3up = {}
        self.line3cent = {}
        self.line3down = {}
        
        self.fig, self.ax = self._init_plot(nfig=nfig)
        self.multi3 = None 
        
        self.plotquantity = plotquantity
        self.unit = unit
        self.normalize = normalizer is not None
        self.normalizer = normalizer
        
        self.wexp = wexp
        self.Iexpcalib = Iexpcalib
        self.wexp_shift = wexp_shift
        
        self.slit = slit
        self.slit_options = slit_options
        
    
    def _init_plot(self, nfig=None):
                
        # Generate Figure 3 layout
        plt.figure(3, figsize=(12,8)).clear()
        fig, ax = plt.subplots(3, 1, num=3, sharex=True)
        plt.tight_layout()

        return fig, ax
    
    
        
    def plot_all_slabs(self, s, slabs):
        
        # Init variables
        fig3 = self.fig
        ax3 = self.ax
        line3cent = self.line3cent
        line3up = self.line3up
        line3down = self.line3down
        
        plotquantity = self.plotquantity
#        unit = self.unit 
        normalize = self.normalize
        norm_on = self.normalizer
        
        wexp = self.wexp
        Iexpcalib = self.Iexpcalib
        wexp_shift = self.wexp_shift
        
        slit = self.slit
        slit_options = self.slit_options
        
        
        # Central axe: model vs experiment
        w, I = s.get(plotquantity)
        ydata = norm_on(w, I) if normalize else I
        try:
            line3cent[1].set_data(w, ydata)
    #        line3cent[1].set_data(*s.get('radiance'))
        except KeyError:
            line3cent[1] = ax3[1].plot(w, ydata, color='r', lw=1, 
                     label='Model')[0]
            ax3[1].set_ylabel(s.units[plotquantity])
            
        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        try:        
            line3cent[0].set_data(wexp+wexp_shift, ydata)
        except KeyError:
            line3cent[0] = ax3[1].plot(wexp+wexp_shift, ydata,'-k', lw=1, zorder=-1, 
                     label='Experiment')[0]
            ax3[1].legend()
            
        def colorserie():
            i = 0
            colorlist = ['r', 'b', 'g', 'y', 'k', 'm']
            while True:
                yield colorlist[i%len(colorlist)]
                i += 1
                
        # Upper axe: emission   &    lower axe: transmittance
        try:
            colors = colorserie()
            for i, (name, s) in enumerate(slabs.items()):
                s.apply_slit(slit, **slit_options)
                color = next(colors)
                line3up[i].set_data(*s.get('radiance'))
                line3down[i].set_data(*s.get('transmittance'))
        except KeyError:  # first time: init lines
            colors = colorserie()
            for i, (name, si) in enumerate(slabs.items()):
                si.apply_slit(slit, **slit_options)
                color = next(colors)
                line3up[i] = ax3[0].plot(*si.get('radiance'), color=color, lw=2, 
                       label=name)[0]
                line3down[i] = ax3[2].plot(*si.get('transmittance'), color=color, lw=2, 
                         label=name)[0]
            if not normalize: 
                ax3[2].set_ylim((-0.008, 0.179)) # Todo: remove that 
            ax3[2].set_ylim((0,1))
            ax3[0].legend()
            ax3[2].legend()
            ax3[0].xaxis.label.set_visible(False)
            ax3[1].xaxis.label.set_visible(False)
            ax3[2].set_xlabel(s.wavespace())
            ax3[0].set_ylabel(si.units['radiance'])
            ax3[2].set_ylabel(si.units['transmittance'])
            fig3.tight_layout()
            
        self._add_multicursor()

        
    def _add_multicursor(self):
        ''' Add vertical bar (if not there already)'''
    
        ax = self.ax

        if self.multi3 is None:
            ax = self.ax 
            multi3 = MultiCursor(self.fig.canvas, (ax[0], ax[1], ax[2]), 
                                 color='r', lw=1, alpha=0.2, horizOn=False, 
                                 vertOn=True)
            self.multi3 = multi3
        else:
            pass
        