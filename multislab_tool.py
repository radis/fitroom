# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:41:43 2017

@author: erwan

Summary
-------

a window to decompose the center slab along the different slabs

-------------------------------------------------------------------------------


"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from mpldatacursor import HighlightingDataCursor
from neq.phys.conv import cm2eV
from neq.plot import plot_stack
from neq.plot.toolbar import add_tools
import warnings
from numpy import nan
from publib import set_style, fix_style


class MultiSlabPlot():

    def __init__(self,
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 normalizer=None,
                 s_exp=None,
                 nfig=None,
                 N_main_bands=5,
                 keep_highlights=False,
                 show_noslit_slabs=True,
                 show_slabs_with_slit=True,
                 ):
        ''' Plot the center case in :class:`~neq.math.fitroom.selection_tool.CaseSelector` 
        by also showing emission and absorption separately, and all slabs 
        
        Parameters
        ----------

        ...

        N_main_bands: int
            show main emission bands in case an overpopulation tool is defined.
            N_main_bands is the number of bands to show. Default 5. 

        keep_highlights: boolean
            if ``True``, delete previous highlights when generating new case. Keeping
            them can help remember the last band position. Default ``False``.

        show_noslit_slabs: boolean
            if ``True``, overlay slabs with non convoluted radiance / transmittance

        show_slabs_with_slit: boolean
            if ``True``, slit is applied to all slabs before display (this does not 
            change the way the radiative transfer equation is solved)


        Examples
        --------
        
        See the working case in :mod:`~neq.test.math.test_fitroom`. In particular, run
        :func:`~neq.test.math.test_fitroom.test_start_fitroom`
            
        See Also
        --------
        
        :class:`~neq.math.fitroom.selection_tool.CaseSelector`,
        :class:`~neq.math.fitroom.grid3x3_tool.Grid3x3`,
        :class:`~neq.math.fitroom.solver.SlabsConfigSolver`,
        :class:`~neq.math.fitroom.noneq_tool.Overpopulator`,
        :class:`~neq.math.fitroom.room.FitRoom`,
        :class:`~neq.math.fitroom.slit_tool.SlitTool` 
        
        '''

        # Init variables
        self.line3up = {}
        self.line3up_noslit = {}
        self.line3cent = {}
        self.line3cent_noslit = {}
        self.line3down = {}
        self.line3down_noslit = {}
        self.line3upbands = {}

        self.fig, self.ax = self._init_plot(nfig=nfig)
        try:
            add_tools()       # includes a Ruler
        except:
            pass
        self.multi3 = None

        self.plotquantity = plotquantity
        self.unit = unit
        self.normalize = normalizer is not None
        self.normalizer = normalizer
        
        self.plot_legend = True   # normally always true

        self.s_exp = s_exp
        if s_exp is not None:
            wexp, Iexpcalib = s_exp.get(plotquantity, Iunit=unit)
        self.wexp = wexp
        self.Iexpcalib = Iexpcalib

        self.N_main_bands = N_main_bands
        self.keep_highlights = keep_highlights

        self.show_noslit_slabs = show_noslit_slabs
        self.show_slabs_with_slit = show_slabs_with_slit

        self.spectrum = None  # hold the current calculated spectrum object

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def _init_plot(self, nfig=None):

        set_style('origin')

        # Generate Figure 3 layout
        plt.figure(3, figsize=(12, 8)).clear()
        fig, ax = plt.subplots(3, 1, num=3, sharex=True)
        plt.tight_layout()

        return fig, ax

    def update(self):
        ''' Get, calculate and plot the current config '''

        slabsconfig = self.fitroom.get_config()
        self.fitroom.eval_dynvar(slabsconfig)  # update dynamic parameters
        calc_slabs = self.fitroom.solver.calc_slabs

        s, slabs, fconfig = calc_slabs(**slabsconfig)

        self.spectrum = s
        self.slabs = slabs
        self.plot_all_slabs(s, slabs)

        self.update_markers(fconfig)

    def update_slit(self):

        slit_function = self.fitroom.solver.slit
        slit_options = self.fitroom.solver.slit_options

        s = self.spectrum
        slabs = self.slabs

        s.apply_slit(slit_function, verbose=False, **slit_options)
        if self.show_slabs_with_slit:
            for sl in slabs.values():
                sl.apply_slit(slit_function, verbose=False, **slit_options)

        self.plot_all_slabs(s, slabs)

    def _plot_failed(self, error_msg):
        line3cent = self.line3cent

        try:
            w, ydata = line3cent[1].get_data()
            line3cent[1].set_data(w*nan, ydata*nan)
            print(error_msg)
        except KeyError:
            pass   # spectrum not plotted yet. Do nothing

        return

    def _format_label(self, label):
        ''' Format with nice output '''

        label = label.replace('m2', 'm$^2$')

        return label

    def plot_for_export(self, style=['origin'],
                        lw_multiplier=1):
        ''' Not used in Fitroom, but can be used by user to export / save figures
        
        Examples
        --------
        
        ::
            
            fig0, fig1 = slabsTool.plot_for_export()
            fig0.savefig('...')
            
        '''

        ylabelsize = 24

        slab_colors = {'sPlasmaCO2': 'b',
                       #                       'sPlasmaCO2b':'cornflowerblue',
                       'sPostCO2': 'g',
                       'sPostCO': 'orange',
                       'sRoomCO2': 'k',
                       'sPlasmaCO': 'r', }

        set_style(style)

        s = self.spectrum
        slabs = self.slabs.copy()

#        # Merge some.. TEMP
#        from radis import SerialSlabs
#        if 'sPlasmaCO2b' in slabs:
#            sPlasmaCO2 = slabs.pop('sPlasmaCO2')
#            sPlasmaCO2b = slabs.pop('sPlasmaCO2b')
#            slabs['sPlasmaCO2'] = SerialSlabs(sPlasmaCO2, sPlasmaCO2b)
#        if 'sPlasmaCOb' in slabs:
#            sPlasmaCO = slabs.pop('sPlasmaCO')
#            sPlasmaCOb = slabs.pop('sPlasmaCOb')
#            slabs['sPlasmaCO'] = SerialSlabs(sPlasmaCO, sPlasmaCOb)

#        plt.figure(figsize=(15,10))
        fig30, ax30 = plt.subplots(figsize=(20, 4))
        fig31, [ax31, ax32] = plt.subplots(2, 1, figsize=(20, 6.5))
#        fig3, ax32 = plt.subplots(figsize=(15,10))
        ax3 = ax31, ax30, ax32

#        plt.figure(figsize=(12,8))
#        fig3, ax3 = plt.subplots(3, 1, num=3, sharex=True)
#        ax3 = ax3[0], ax3[2], ax3[1]

#        if s is None:
#            self._plot_failed(error_msg='Slabs tool: spectrum could not be calculated')
#            return

        # Init variables

        plotquantity = self.plotquantity
        unit = self.unit
        normalize = self.normalize
        norm_on = self.normalizer

        wexp = self.wexp
        Iexpcalib = self.Iexpcalib

        slit = self.fitroom.solver.slit
        slit_options = self.fitroom.solver.slit_options

        # Central axe: model vs experiment
        w, I = s.get(plotquantity, Iunit=unit)
        ydata = norm_on(w, I) if normalize else I

        ax3[1].plot(w, ydata, color='r', lw=1*lw_multiplier,
                    label='Model')[0]
#        if self.show_noslit_slabs and not normalize:
#            ymax = ax3[1].get_ylim()[1]
#            ax3[1].plot(*s.get(plotquantity+'_noslit', Iunit=unit),
#                            color='r', lw=0.5*lw_multiplier, alpha=0.15, zorder=-1)[0]
#            ax3[1].set_ylim(ymax=ymax)  # keep no slit yscale
        ax3[1].set_ylabel(self._format_label(unit), size=ylabelsize)

        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib

        plot_stack(wexp, ydata, '-k',
                   lw=1*lw_multiplier, zorder=-1, label='Experiment', ax=ax3[1])[0]
        if self.plot_legend:
            ax3[1].legend()

        def colorserie():
            i = 0
            colorlist = ['r', 'b', 'g', 'y', 'k', 'm']
            while True:
                yield colorlist[i % len(colorlist)]
                i += 1

        # Upper axe: emission   &    lower axe: transmittance
        colors = colorserie()
        for i, (name, si) in enumerate(slabs.items()):
            si.apply_slit(slit, verbose=False, **slit_options)
            if name in slab_colors:
                color = slab_colors[name]
            else:
                color = next(colors)

            ls = '-' if i < 6 else '--'
            ax3[0].plot(*si.get('radiance', Iunit=unit), color=color,
                        lw=1*lw_multiplier, ls=ls, label=name.replace('sP', 'P').replace('sR', 'R'))[0]
            ax3[2].plot(*si.get('transmittance'), color=color,
                        lw=1*lw_multiplier, ls=ls, label=name.replace('sP', 'P').replace('sR', 'R'))[0]
        if self.show_noslit_slabs:
            colors = colorserie()
            ymax = ax3[0].get_ylim()[1]
            for i, (name, si) in enumerate(slabs.items()):
                if name in slab_colors:
                    color = slab_colors[name]
                else:
                    color = next(colors)
                ax3[0].plot(*si.get('radiance_noslit', Iunit=unit), color=color,
                            lw=0.5*lw_multiplier, ls=ls, alpha=0.15, zorder=-1)[0]
#                    ax3[0].set_ylim(ymax=ymax)  # keep no slit yscale
                ax3[2].plot(*si.get('transmittance_noslit'), color=color,
                            lw=0.5*lw_multiplier, ls=ls, alpha=0.15, zorder=-1)[0]
            ax3[0].set_ylim(ymax=ymax)  # keep no slit yscale
#            if not normalize:
#                ax3[2].set_ylim((-0.008, 0.179)) # Todo: remove that
        ax3[2].set_ylim((0, 1))
#            ax3[2].legend()
        ax3[2].set_xlabel('Wavelength (nm)')
        ax3[1].set_xlabel('Wavelength (nm)')
        ax3[0].set_ylabel(self._format_label(unit), size=ylabelsize)
        ax3[2].set_ylabel(self._format_label(
            si.units['transmittance']), size=ylabelsize)

        # Main bands
        self._add_bands()

#        # Cursors
#        self._add_multicursor()

        for ax in ax3:
#            ax.set_xlim((4150, 4850))
            fix_style(style=style, ax=ax, tight_layout=False)

        if self.plot_legend:
            ax3[0].legend(loc='lower right', bbox_to_anchor=[
                          1.0, 0.2], prop={'size': 24}, ncol=2)

        ax3[0].xaxis.label.set_visible(False)
#        ax3[1].xaxis.label.set_visible(True)
        ax3[2].xaxis.label.set_visible(True)
#        ax3[2].tick_params(labelbottom='off')
        ax3[0].tick_params(labelbottom='off')

        fig30.tight_layout()
        fig31.tight_layout()
        fig31.subplots_adjust(hspace=0)

        return fig30, fig31

    def plot_all_slabs(self, s, slabs):

        if s is None:
            self._plot_failed(
                error_msg='Slabs tool: spectrum could not be calculated')
            return

        # Init variables
        fig3 = self.fig
        ax3 = self.ax
        line3cent = self.line3cent
        line3cent_noslit = self.line3cent_noslit
        line3up = self.line3up
        line3up_noslit = self.line3up_noslit
        line3down = self.line3down
        line3down_noslit = self.line3down_noslit

        plotquantity = self.plotquantity
        unit = self.unit
        normalize = self.normalize
        norm_on = self.normalizer

        wexp = self.wexp
        Iexpcalib = self.Iexpcalib

        slit = self.fitroom.solver.slit
        slit_options = self.fitroom.solver.slit_options

        # Central axe: model vs experiment
        w, I = s.get(plotquantity, Iunit=unit)
        ydata = norm_on(w, I) if normalize else I
        try:
            line3cent[1].set_data(w, ydata)
    #        line3cent[1].set_data(*s.get('radiance'))
            if self.show_noslit_slabs and not normalize:
                line3cent_noslit[1].set_data(
                    *s.get(plotquantity+'_noslit', Iunit=unit))
        except KeyError:
            line3cent[1] = ax3[1].plot(w, ydata, color='r', lw=0.5,
                                       label='Model')[0]
#            if self.show_noslit_slabs and not normalize:
#                ymax = ax3[1].get_ylim()[1]
#                line3cent_noslit[1] = ax3[1].plot(*s.get(plotquantity+'_noslit', Iunit=unit),
#                                color='r', lw=0.5, alpha=0.15, zorder=-1)[0]
#                ax3[1].set_ylim(ymax=ymax)  # keep no slit yscale
            ax3[1].set_ylabel(self._format_label(unit))

        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        try:
            line3cent[0]  # doesnt change  .set_data(wexp, ydata)
        except KeyError:
            line3cent[0] = plot_stack(wexp, ydata, '-k',
                                      lw=0.5, zorder=-1, label='Experiment', ax=ax3[1])[0]
            if self.plot_legend:
                ax3[1].legend()

        def colorserie():
            i = 0
            colorlist = ['r', 'b', 'g', 'y', 'k', 'm']
            while True:
                yield colorlist[i % len(colorlist)]
                i += 1

        # Upper axe: emission   &    lower axe: transmittance
        try:
            colors = colorserie()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', "interpolating slit function over spectrum grid")
                for i, (name, s) in enumerate(slabs.items()):
                    s.apply_slit(slit, verbose=False, **slit_options)
                    color = next(colors)
                    line3up[i].set_data(*s.get('radiance', Iunit=unit))
                    line3down[i].set_data(*s.get('transmittance'))
                    if self.show_noslit_slabs:
                        line3up_noslit[i].set_data(
                            *s.get('radiance_noslit', Iunit=unit))
                        line3down_noslit[i].set_data(
                            *s.get('transmittance_noslit'))
        except KeyError:  # first time: init lines
            colors = colorserie()
            for i, (name, si) in enumerate(slabs.items()):
                si.apply_slit(slit, verbose=False, **slit_options)
                color = next(colors)
                ls = '-' if i < 6 else '--'
                line3up[i] = ax3[0].plot(*si.get('radiance', Iunit=unit), color=color,
                                         lw=0.5, ls=ls, label=name)[0]
                line3down[i] = ax3[2].plot(*si.get('transmittance'), color=color,
                                           lw=0.5, ls=ls, label=name)[0]
            if self.show_noslit_slabs:
                colors = colorserie()
                ymax = ax3[0].get_ylim()[1]
                for i, (name, si) in enumerate(slabs.items()):
                    color = next(colors)
                    line3up_noslit[i] = ax3[0].plot(*si.get('radiance_noslit', Iunit=unit), color=color,
                                                    lw=0.5, ls=ls, alpha=0.15, zorder=-1)[0]
#                    ax3[0].set_ylim(ymax=ymax)  # keep no slit yscale
                    line3down_noslit[i] = ax3[2].plot(*si.get('transmittance_noslit'), color=color,
                                                      lw=0.5, ls=ls, alpha=0.15, zorder=-1)[0]
                ax3[0].set_ylim(ymax=ymax)  # keep no slit yscale
#            if not normalize:
#                ax3[2].set_ylim((-0.008, 0.179)) # Todo: remove that
            ax3[2].set_ylim((0, 1))
            if self.plot_legend:
                ax3[0].legend()
                ax3[2].legend()
            ax3[0].xaxis.label.set_visible(False)
            ax3[1].xaxis.label.set_visible(False)
            ax3[2].set_xlabel('Wavelength (nm)')
            ax3[0].set_ylabel(self._format_label(unit))
            ax3[2].set_ylabel(self._format_label(si.units['transmittance']))
            fig3.tight_layout()

        # Main bands
        self._add_bands()

        # Cursors
        self._add_multicursor()

        fix_style(tight_layout=False)

        self.fig.canvas.draw()

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

# Clean previous highlights
        if not self.keep_highlights:
            self._clean_highlights()

        # Deactivated feature for the moment

#        # Add main bands manually
#        ax0 = self.ax[0]
#        lines = []
#        for lvl, E in overpTool.E_lvls.items(): #lvlist.sorted_bands[:self.N_main_bands]:
#            sb = overpTool.bandlist.bands[br]
#            sb.apply_slit(slit, energy_threshold=0.2)
##            get_rad = 'radiance' if self.slit_on_slabs else 'radiance_noslit'
#            w, I = sb.get('radiance', Iunit=unit)
#            if br in list(line3upbands.keys()):
#                line3upbands[br].set_data(w, I)
#                lines.append(line3upbands[br])
#            else:
#                l, = ax0.plot(w, I, color='grey', label='{0} ({1:.2f}eV)'.format(br,
#                              cm2eV(overpTool.bandlist.E_bands[br])))
#                line3upbands[br] = l
#                lines.append(l)
#        self.highlights = HighlightingDataCursor(lines)

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

    def format_coord(self, x, y):
        # TODO: implement it, but i cant remember where... Lookup grid_tool
        return 'x = {0:.2f} nm, y = {1:.4f} {2}  '.format(x, y, self.unit)

    def update_markers(self, fconfig):

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        if self.fitroom.selectTool is not None:
            self.fitroom.selectTool.update_markers(fconfig)
            self.fitroom.selectTool.fig.canvas.draw()

        else:
            print('... No case selector defined')
        return
