# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:38:25 2017

@author: erwan

# Todo: Overlay with "template" slit (ex: imported )

-------------------------------------------------------------------------------
"""


from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from radis.tools.slit import (get_slit_function, get_effective_FWHM, get_FWHM)
from radis.spectrum.utils import WAVELEN_UNITS, WAVENUM_UNITS
from radis.phys.convert import nm2cm, cm2nm
from radis.misc.basics import is_float
import numpy as np
from six import string_types
import warnings
from warnings import warn


class SlitTool():
    ''' Tool to manipulate slit function 
    
    Parameters
    ----------

    plot_unit: 'nm', 'cm-1'

    overlay: str, or int, or tuple
        plot a slit in background (to compare generated slit with experimental 
        slit for instance)
        
    Examples
    --------
    
    .. minigallery:: fitroom.SlitTool

    See Also
    --------
    
    :class:`~fitroom.selection_tool.CaseSelector`,
    :class:`~fitroom.grid3x3_tool.Grid3x3`,
    :class:`~fitroom.multislab_tool.MultiSlabPlot`,
    :class:`~fitroom.solver.SlabsConfigSolver`,
    :class:`~fitroom.noneq_tool.Overpopulator`,
    :class:`~fitroom.room.FitRoom`
    
    '''
    def __init__(self, plot_unit='nm', overlay=None, overlay_options=None):

        self.fitroom = None               # type: FitRoom
        self.plot_unit = plot_unit
        self.overlay = overlay
        self.overlay_options = overlay_options

        plt.figure('SlitTool').clear()
        fig, ax = plt.subplots(num='SlitTool')
        self.fig = fig
        self.ax = ax
        self.lines = {}

        # for sliders:
        self.sltop = None       # type: Slider
        self.slbase = None      # type: Slider

    def slit_function(self):
        return self.fitroom.solver.slit

    def slit_options(self):
        return self.fitroom.solver.slit_options

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

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
                raise ValueError(
                    'Wrong slit function format: {0}'.format(slit_function))

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

        if not hasattr(self, 'fitroom'):
            return

        slabsTool = self.fitroom.slabsTool
        gridTool = self.fitroom.gridTool
        slit_function = self.slit_function()
        plot_unit = self.plot_unit

        # Get one spectrum  to get wavespace range
        if slabsTool is not None:
            if slabsTool.spectrum is not None:
                s = slabsTool.spectrum
            else:
                return
        elif gridTool is not None:
            try:
                s = gridTool.spectra[(0, 0)]
            except KeyError:
                return
        else:  # do nothing
            return

        slit_unit = s.conditions['slit_unit']
        wstep = s.conditions['wstep']
        norm_by = s.conditions['norm_by']

        slit_options = self.slit_options()
        shape = slit_options['shape']

        if 'center_wavespace' in slit_options:
            center_wavespace = slit_options['center_wavespace']
        else:
            # center_wavespace is inferred by apply_slit() from the slit range:
            # we do the same here
            waveunit = s.get_waveunit()
            w = s._q['wavespace']
            # reminder: center_wavespace should be in ~ unit
            center_wavespace = w[len(w)//2]   # w ~ waveunit
            if waveunit in WAVENUM_UNITS and slit_unit in WAVELEN_UNITS:
                center_wavespace = cm2nm(
                    center_wavespace)     # wavenum > wavelen
            elif waveunit in WAVELEN_UNITS and slit_unit in WAVENUM_UNITS:
                center_wavespace = nm2cm(
                    center_wavespace)     # wavelen > wavenum

        # Plot
#        slit_options = self.slit_options

        wslit, Islit = get_slit_function(slit_function, unit=slit_unit, return_unit=plot_unit,
                                         norm_by=norm_by,
                                         center_wavespace=center_wavespace, wstep=wstep,
                                         shape=shape, plot=False)

        woverlay, Ioverlay = None, None

        if self.overlay is not None:
            overlay_options = self.overlay_options

            if overlay_options is None:
                overlay_options = {}
            ov_slit_unit = overlay_options.get('unit', 'nm')
            if 'center_wavespace' in overlay_options:
                ov_center_wavespace = overlay_options['center_wavespace']
            else:
                waveunit = s.get_waveunit()
                w = s._q['wavespace']
                # reminder: center_wavespace should be in ~ unit
                ov_center_wavespace = w[len(w)//2]  # w ~ waveunit
                if waveunit in WAVENUM_UNITS and ov_slit_unit in WAVELEN_UNITS:
                    ov_center_wavespace = cm2nm(
                        center_wavespace)     # wavenum > wavelen
                elif waveunit in WAVELEN_UNITS and ov_slit_unit in WAVENUM_UNITS:
                    ov_center_wavespace = nm2cm(
                        center_wavespace)     # wavelen > wavenum

            ov_wstep = s.conditions['wstep']
            ov_norm_by = overlay_options.get('norm_by', 'area')
            ov_shape = overlay_options.get('shape', 'triangular')

            woverlay, Ioverlay = get_slit_function(self.overlay, unit=ov_slit_unit, return_unit=plot_unit,
                                                   center_wavespace=ov_center_wavespace,
                                                   norm_by=ov_norm_by, wstep=ov_wstep, shape=ov_shape,
                                                   plot=False)

        self.plot_slit(wslit, Islit, waveunit=plot_unit, plot_unit=plot_unit,
                       wover=woverlay, Iover=Ioverlay)

    def plot_slit(self, w, I=None, waveunit='', plot_unit='same',
                  wover=None, Iover=None):
        ''' Variant of the plot_slit function defined in slit.py that can 
        set_data when figure already exists

        Plot slit, calculate and display FWHM, and calculate effective FWHM.
        FWHM is calculated from the limits of the range above the half width,
        while FWHM is the equivalent width of a triangular slit with the same area

        Parameters
        ----------

        w, I: arrays    or   (str, None)
            if str, open file directly

        waveunit: 'nm', 'cm-1' or ''

        plot_unit: 'nm, 'cm-1' or 'same'
            change plot unit (and FWHM units)

        warnings: boolean
            if ``True``, test if slit is correctly centered and output a warning if it
            is not. Default ``True``

        '''

        fig = self.fig
        ax = self.ax

        try:
            from radis.tools.plot_tools import add_ruler
        except ImportError:
            pass

        # Check input
        if isinstance(w, string_types) and I is None:
            w, I = np.loadtxt(w).T
        assert len(w) == len(I)
        if np.isnan(I).sum() > 0:
            warn('Slit function has nans')
            w = w[~np.isnan(I)]
            I = I[~np.isnan(I)]
        assert len(I) > 0
        if waveunit not in WAVELEN_UNITS+WAVENUM_UNITS:
            raise ValueError('Unknown wavespace unit: {0}'.format(waveunit))

        # Recalculate FWHM
        FWHM, _, _ = get_FWHM(w, I, return_index=True)
        FWHM_eff = get_effective_FWHM(w, I)

        # Convert wavespace unit if needed
        if plot_unit == 'same':
            pass
        elif plot_unit not in WAVELEN_UNITS+WAVENUM_UNITS:
            raise ValueError('Unknown wavespace unit: {0}'.format(plot_unit))
        elif waveunit in WAVENUM_UNITS and plot_unit in WAVELEN_UNITS:  # wavelength > wavenumber
            w = cm2nm(w)
            waveunit = 'nm'
        elif waveunit in WAVELEN_UNITS and plot_unit in WAVENUM_UNITS:  # wavenumber > wavelength
            w = nm2cm(w)
            waveunit = 'cm-1'
        else:  # same
            pass
#            raise ValueError('Unknown plot unit: {0}'.format(plot_unit))

        xlabel = 'Wavespace'
        if waveunit in WAVELEN_UNITS:
            xlabel = 'Wavelength (nm)'
            unit = ' nm'
        elif waveunit in WAVENUM_UNITS:
            xlabel = 'Wavenumber (cm-1)'
            unit = ' cm-1'
        elif waveunit == '':
            unit = ''
        else:
            raise ValueError('Unknown wavespace unit: {0}'.format(waveunit))

#        w[len(w)//2]
        ax.set_title(
            'FWHM {0:.2f}nm, effective {1:.2f}nm'.format(FWHM, FWHM_eff))

        try:
            self.lines[0].set_data(w, I)
            self.lines[1].set_data(w, I)
            ax.relim()
            ax.autoscale_view()
        except KeyError:
            self.lines[0], = ax.plot(w, I, 'o', color='lightgrey')
            self.lines[1], = ax.plot(w, I, '-k', label='FWHM: {0:.3f} {1}'.format(FWHM, unit) +
                                     '\nEff. FWHM: {0:.3f} {1}'.format(FWHM_eff, unit) +
                                     '\nArea: {0:.3f}'.format(
                                         abs(np.trapz(I, x=w))),
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
#            plt.legend(loc='best', prop={'size':16})

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
        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        self.fitroom.solver.slit = slit_function
        print(('New slit assigned:', slit_function))

        # replot
        slabsTool = self.fitroom.slabsTool
        gridTool = self.fitroom.gridTool

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', "Triangular slit given with a tuple")

            if slabsTool is not None:
                slabsTool.update_slit()
            if gridTool is not None:
                gridTool.update_slit()
#                gridTool.fig.show() # or something smarter.  # TODO # HERE

        self.update_figure()


#    def update_slit():
#
