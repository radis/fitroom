# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:38:25 2017

@author: erwan

Todo
-------

Overlay with "template" slit (ex: imported )

"""


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from neq.spec.slit import (get_slit_function, WAVELEN_UNITS, WAVENUM_UNITS,
                           get_effective_FWHM, get_FWHM)
from neq.phys.conv import nm2cm, cm2nm
from neq.misc import is_float
import numpy as np
from six import string_types
import warnings
from warnings import warn


class SlitTool():
    ''' Tool to manipulate slit function '''
    
    def __init__(self, plot_unit='nm', overlay=None, overlay_options=None):
        ''' 
        Input
        --------
        
        plot_unit: 'nm', 'cm-1'
        
        overlay: str, or int, or tuple
            plot a slit in background (to compare generated slit with experimental 
            slit for instance)
        '''

        self.fitroom = None               # type: FitRoom
        self.plot_unit = plot_unit
        self.overlay = overlay
        self.overlay_options = overlay_options
        
        plt.figure('SlitTool').clear()
        fig, ax = plt.subplots(num='SlitTool')
        self.fig = fig
        self.ax = ax 
        self.lines = {}
        
    def slit_function(self):
        return self.fitroom.solver.slit
    
    def slit_options(self):
        return self.fitroom.solver.slit_options
    
    def connect(self):
        ''' Triggered on connection to FitRoom '''

        slit_function = self.slit_function()
#        slit_options = self.slit_options()
        
        if not isinstance(slit_function, string_types):
            plt.subplots_adjust(left=0.15, bottom=0.25, top=0.9)
            
            # Add sliders
            if is_float(slit_function):
                base = slit_function
                top = 0
            elif isinstance(slit_function, tuple):
                base, top = max(slit_function), min(slit_function)
            else:
                raise ValueError('Wrong slit function format: {0}'.format(slit_function))
                        
#            plt.axis([0, 1, -10, 10])
            
            axcolor = 'lightgoldenrodyellow'
            axtop = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
            axbase = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

            sltop = Slider(axtop, 'Top', 0, 3*base, valinit=top)
            slbase = Slider(axbase, 'Base', 0, 3*base, valinit=base)
    
            sltop.on_changed(self.update_slider)
            slbase.on_changed(self.update_slider)
            
            self.sltop = sltop
            self.slbase = slbase
            
        self.update_figure()
        
    def update_figure(self):
        
        if self.fitroom is None:
            return
        
        slabsTool = self.fitroom.slabsTool
        gridTool = self.fitroom.gridTool
        plotquantity = self.fitroom.solver.plotquantity
        slit_function = self.slit_function()
        plot_unit = self.plot_unit
        
        slit_options = self.slit_options()
        
        # Get one spectrum  to get wavespace range
        if slabsTool is not None:
            if slabsTool.spectrum is not None:
                s = slabsTool.spectrum
            else:
                return
        elif gridTool is not None:
            try:
                s = gridTool.spectra[(0,0)]
            except KeyError:
                return 
        else:  # do nothing
            return  
        
        if 'center_wavespace' in slit_options:
            center_wavespace = slit_options['center_wavespace']
        else:
            waveunit = s.get_waveunit()
            w, I = s.get(plotquantity, wunit=waveunit)
            center_wavespace = w[len(w)//2]
        wstep = s.conditions['wstep']
        norm_by = slit_options.get('norm_by', 'area')
        shape = slit_options.get('shape', 'triangular')
        slit_unit = slit_options.get('slit_unit', 'nm')

        # Plot
#        slit_options = self.slit_options 
        
        wslit, Islit = get_slit_function(slit_function, waveunit, center_wavespace,
                                         norm_by, wstep, shape, slit_unit, plot=False)
        
        if self.overlay is not None:
            overlay_options = self.overlay_options
            if overlay_options is None:
                overlay_options = {}
            if 'center_wavespace' in overlay_options:
                over_center_wavespace = overlay_options['center_wavespace']
            else:
                over_center_wavespace = w[len(w)//2]
            over_wavespace = overlay_options.get('wavespace', 'nm')
            over_wstep = s.conditions['wstep']
            over_norm_by = overlay_options.get('norm_by', 'area')
            over_shape = overlay_options.get('shape', 'triangular')
            over_slit_unit = overlay_options.get('slit_unit', 'nm')

            woverlay, Ioverlay = get_slit_function(self.overlay, over_wavespace, 
                                                   over_center_wavespace,
                                         over_norm_by, over_wstep, over_shape, 
                                         over_slit_unit, plot=False)
        
        self.plot_slit(wslit, Islit, waveunit=waveunit, plot_unit=plot_unit, 
                       wover=woverlay, Iover=Ioverlay)
            
    def plot_slit(self, w, I=None, waveunit='', plot_unit='same',
                  wover=None, Iover=None):
        ''' Variant of the plot_slit functino defined in slit.py that can 
        set_data when figure already exists
        
        Plot slit, calculate and display FWHM, and calculate effective FWHM.
        FWHM is calculated from the limits of the range above the half width,
        while FWHM is the equivalent width of a triangular slit with the same area
    
        Input
        --------
    
        w, I: arrays    or   (str, None)
            if str, open file directly
    
        wavespace: 'nm', 'cm-1' or ''
    
        plot_unit: 'nm, 'cm-1' or 'same'
            change plot unit (and FWHM units)
    
        warnings: boolean
            if True, test if slit is correctly centered and output a warning if it
            is not. Default True
    
        '''
        
        fig= self.fig
        ax= self.ax
        
        from neq.plot.toolbar import add_tools
        try:
            add_tools()       # includes a Ruler to measure slit 
        except:
            pass
        
        # Check input
        if isinstance(w, string_types) and I is None:
            w, I = np.loadtxt(w).T
        assert len(w)==len(I)
        if np.isnan(I).sum()>0:
            warn('Slit function has nans')
            w = w[~np.isnan(I)]
            I = I[~np.isnan(I)]
        assert len(I)>0
            
        # Recalculate FWHM
        FWHM, xmin, xmax = get_FWHM(w, I, return_index=True)
        FWHM_eff = get_effective_FWHM(w, I)
    
        # Convert wavespace unit if needed
        if plot_unit == 'same':
            pass
        elif plot_unit in WAVELEN_UNITS+WAVENUM_UNITS and waveunit not in WAVENUM_UNITS+WAVENUM_UNITS:
            raise ValueError('Unknown wavespace unit: {0}'.format(waveunit))
        elif waveunit in WAVENUM_UNITS and plot_unit in WAVELEN_UNITS: # wavelength > wavenumber
            w = cm2nm(w)
            wavespace = 'nm'
        elif wavespace in WAVELEN_UNITS and plot_unit in WAVENUM_UNITS: # wavenumber > wavelength
            w = nm2cm(w)
            wavespace = 'cm-1'
        else:
            raise ValueError('Unknown plot unit: {0}'.format(plot_unit))
    
        xlabel = 'Wavespace'
        if wavespace in WAVELEN_UNITS:
            xlabel = 'Wavelength (nm)'
            unit = ' nm'
        elif wavespace in WAVENUM_UNITS:
            xlabel = 'Wavenumber (cm-1)'
            unit = ' cm-1'
        elif wavespace == '':
            unit = ''
        else:
            raise ValueError('Unkown wavespace: {0}'.format(wavespace))
    
#        w[len(w)//2]
        ax.set_title('FWHM {0:.2f}nm, effective {1:.2f}nm'.format(FWHM, FWHM_eff))
        
        try:
            self.lines[0].set_data(w,I)
            self.lines[1].set_data(w,I)
            ax.relim()
            ax.autoscale_view()
        except KeyError:
            self.lines[0], = ax.plot(w, I, 'o', color='lightgrey')
            self.lines[1], = ax.plot(w, I, '-k', label='FWHM: {0:.3f} {1}'.format(FWHM, unit)+\
                                     '\nEff. FWHM: {0:.3f} {1}'.format(FWHM_eff, unit)+\
                                     '\nArea: {0:.3f}'.format(abs(np.trapz(I, x=w))),
                                     )
        
            # TODO: reactivate with set_data
    #        # Vertical lines on center, and FWHM
    #        plt.axvline(w[len(w)//2], ls='-', lw=2, color='lightgrey')  # center
    #        plt.axvline(w[(xmin+xmax)//2], ls='--', color='k', lw=0.5)   # maximum (should be center)
    #        plt.axvline(w[xmin], ls='--', color='k', lw=0.5)      # FWHM min
    #        plt.axvline(w[xmax], ls='--', color='k', lw=0.5)      # FWHM max
    #        plt.axhline(I.max()/2, ls='--', color='k', lw=0.5)      # half maximum
            
            # Add overlay
            if wover is not None and Iover is not None:
                ax.plot(wover, Iover, '-', color='lightgrey')
    
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Slit function')
            plt.legend(loc='best', prop={'size':16})
        
        #   plt.tight_layout()
        
        return

    def update_slider(self, val):
        ''' update slit function in Solver and replot  '''
        
        top = self.sltop.val
        base = self.slbase.val
#        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#        fig.canvas.draw_idle()

        slit_function = (top, base)
        
        # Update default         
        if self.fitroom is None:
            raise ValueError('Fitroom not connected')
        self.fitroom.solver.slit = slit_function
        print('New slit assigned:', slit_function)
        
        
        # replot
        slabsTool = self.fitroom.slabsTool
        gridTool = self.fitroom.gridTool

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "Triangular slit given with a tuple")
            
            if slabsTool is not None:
                slabsTool.update_slit()
            if gridTool is not None:
                gridTool.update_slit()
#                gridTool.fig.show() # or something smarter.  # TODO # HERE 
                
                
        self.update_figure()
        

#    def update_slit():
#        
        