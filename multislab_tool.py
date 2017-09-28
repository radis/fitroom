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
from mpldatacursor import HighlightingDataCursor
from neq.phys.conv import cm2eV
from neq.plot import plot_stack

class MultiSlabPlot():
    
    def __init__(self, 
                 plotquantity='radiance', unit= 'mW/cm2/sr/nm', 
                 normalizer=None,
                 wexp=None, Iexpcalib=None, wexp_shift=0,
                 nfig=None,
                 N_main_bands = 5,
                 keep_highlights = False,
                 ):
        ''' 
        Input
        -----
        
        ...
        
        N_main_bands: int
            show main emission bands in case an overpopulation tool is defined.
            N_main_bands is the number of bands to show. Default 5. 
        
        keep_highlights: boolean
            if True, delete previous highlights when generating new case. Keeping
            them can help remember the last band position. Default False.
        
        '''
        
        # Init variables
        self.line3up = {}
        self.line3cent = {}
        self.line3down = {}
        self.line3upbands = {}  
        
        self.fig, self.ax = self._init_plot(nfig=nfig)
        self.multi3 = None 
        
        self.plotquantity = plotquantity
        self.unit = unit
        self.normalize = normalizer is not None
        self.normalizer = normalizer
        
        self.wexp = wexp
        self.Iexpcalib = Iexpcalib
        self.wexp_shift = wexp_shift
        
        self.fitroom = None
        
        self.N_main_bands = N_main_bands 
        self.keep_highlights = keep_highlights  
        
    
    def _init_plot(self, nfig=None):
                
        # Generate Figure 3 layout
        plt.figure(3, figsize=(12,8)).clear()
        fig, ax = plt.subplots(3, 1, num=3, sharex=True)
        plt.tight_layout()

        return fig, ax
    
    
    def update(self):
        ''' Get, calculate and plot the current config '''
        
        slabsconfig = self.fitroom.get_config()
        
        calc_slabs = self.fitroom.solver.calc_slabs

        s, slabs, fconfig = calc_slabs(**slabsconfig)
        
        self.plot_all_slabs(s, slabs)
        
        self.update_markers(fconfig)
        
    def plot_all_slabs(self, s, slabs):
        
        # Init variables
        fig3 = self.fig
        ax3 = self.ax
        line3cent = self.line3cent
        line3up = self.line3up
        line3down = self.line3down
        
        plotquantity = self.plotquantity
        unit = self.unit 
        normalize = self.normalize
        norm_on = self.normalizer
        
        wexp = self.wexp
        Iexpcalib = self.Iexpcalib
        wexp_shift = self.wexp_shift
        
        slit = self.fitroom.solver.slit
        slit_options = self.fitroom.solver.slit_options
        
        # Central axe: model vs experiment
        w, I = s.get(plotquantity, yunit=unit)
        ydata = norm_on(w, I) if normalize else I
        try:
            line3cent[1].set_data(w, ydata)
    #        line3cent[1].set_data(*s.get('radiance'))
        except KeyError:
            line3cent[1] = ax3[1].plot(w, ydata, color='r', lw=1, 
                     label='Model')[0]
            ax3[1].set_ylabel(unit)
            
        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        try:        
            line3cent[0]  # doesnt change  .set_data(wexp+wexp_shift, ydata)
        except KeyError:
            plt.sca(ax3[1])
            line3cent[0] = plot_stack(wexp+wexp_shift, ydata,'-k', lw=1, zorder=-1, 
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
                line3up[i].set_data(*s.get('radiance', yunit=unit))
                line3down[i].set_data(*s.get('transmittance'))
        except KeyError:  # first time: init lines
            colors = colorserie()
            for i, (name, si) in enumerate(slabs.items()):
                si.apply_slit(slit, **slit_options)
                color = next(colors)
                line3up[i] = ax3[0].plot(*si.get('radiance', yunit=unit), color=color, lw=2, 
                       label=name)[0]
                line3down[i] = ax3[2].plot(*si.get('transmittance'), color=color, lw=2, 
                         label=name)[0]
#            if not normalize: 
#                ax3[2].set_ylim((-0.008, 0.179)) # Todo: remove that 
            ax3[2].set_ylim((0,1))
            ax3[0].legend()
            ax3[2].legend()
            ax3[0].xaxis.label.set_visible(False)
            ax3[1].xaxis.label.set_visible(False)
            ax3[2].set_xlabel('Wavelength (nm)')
            ax3[0].set_ylabel(unit)
            ax3[2].set_ylabel(si.units['transmittance'])
            fig3.tight_layout()
            
        # Main bands
        self._add_bands()
            
        # Cursors
        self._add_multicursor()


    def _clean_datacursors(self):
        
        try:
            self.fig._mpldatacursors
        except AttributeError:
            # no _mpldatacursors, go ahead
            return

        # Disable all         
        for d in self.fig._mpldatacursors:
            d.hide().disable()
            del d 
        
        self.fig._mpldatacursors = []
        
    def _clean_highlights(self):
        
        try:
            self.highlights
        except AttributeError:
            # no _mpldatacursors, go ahead
            return
        
        # Delete previous highlights
        for artist in self.highlights.highlights:
            l = self.highlights.highlights[artist]
            l.set_visible(False)
            l.remove()

    def _add_bands(self):
        
        if self.fitroom.overpTool is None:
            return
        else:
            overpTool = self.fitroom.overpTool
        slit = self.fitroom.solver.slit 
        unit = self.fitroom.solver.unit
        
#        # Delete previous bands
        line3upbands = self.line3upbands 
#        if line3upbands is not None:
#            for l in line3upbands:
#                del l 
                
        # Clean previous datacursors
        self._clean_datacursors()
        
##        # Clean previous highlights
        if not self.keep_highlights:
            self._clean_highlights()
                
        # Add main bands manually
        ax0 = self.ax[0]
        lines = []
        for br in overpTool.bandlist.sorted_bands[:self.N_main_bands]:
            sb = overpTool.bandlist.bands[br]
            sb.apply_slit(slit, energy_threshold=0.2)
            w, I = sb.get('radiance', yunit=unit)
            if br in line3upbands.keys():
                line3upbands[br].set_data(w, I)
                lines.append(line3upbands[br])
            else:
                l, = ax0.plot(w, I, color='grey', label='{0} ({1:.2f}eV)'.format(br, 
                              cm2eV(overpTool.bandlist.E_bands[br])))
                line3upbands[br] = l
                lines.append(l)
        self.highlights = HighlightingDataCursor(lines)
        
        # store lines (so they dont disappear)
        self.line3upbands = line3upbands

        
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
        
        
    def update_markers(self, fconfig):

        if self.fitroom is None:
            raise ValueError('Tool not connected to Fitroom')
        if self.fitroom.selectTool is not None:
            self.fitroom.selectTool.update_markers(fconfig)
            self.fitroom.selectTool.fig.canvas.draw()
            
        else:
            print('... No case selector defined')    
        return
