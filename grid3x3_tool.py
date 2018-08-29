# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:22:30 2017

@author: erwan

Tool to plot on 9 graphes along 2 axes (2 conditions)

-------------------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import textwrap
from matplotlib.widgets import MultiCursor
from neq.plot import plot_stack
from neq.plot.toolbar import add_tools
from neq.plot.pretty import make_up
from neq.spec import Spectrum   # for IDE hints
import warnings
from numpy import nan
from publib import set_style, fix_style


class Grid3x3():

    def __init__(self, slbInteractx=None, slbInteracty=None,
                 xparam='', yparam='',
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 normalizer=None,
                 s_exp=None
                 ):
        ''' Where the output of a :class:`~neq.math.fitroom.selection_tool.CaseSelector` 
        is shown. 
        
        Examples
        --------
        
        See the working case in :mod:`~neq.test.math.test_fitroom`. In particular, run
        :func:`~neq.test.math.test_fitroom.test_start_fitroom`
            
        See Also
        --------
        
        :class:`~neq.math.fitroom.selection_tool.CaseSelector`,
        :class:`~neq.math.fitroom.multislab_tool.MultiSlabPlot`,
        :class:`~neq.math.fitroom.solver.SlabsConfigSolver`,
        :class:`~neq.math.fitroom.noneq_tool.Overpopulator`,
        :class:`~neq.math.fitroom.room.FitRoom`,
        :class:`~neq.math.fitroom.slit_tool.SlitTool` 
        
        When you're happy with a fit, use:
            
        :meth:`~neq.math.fitroom.grid3x3_tool.plot_for_export`
           
        '''

        plt.figure(2, figsize=(16, 12)).clear()
        fig2, ax2 = plt.subplots(3, 3, sharex=True, sharey=True,
                                 num=2)

        self.fig = fig2
        self.ax = ax2
        try:
            add_tools()       # includes a Ruler
        except:
            pass
        self.multi2 = None  # used to save the multicursor afterwards

        self.lineexp = {}
        self.linesim = {}
        self.legends2 = {}
        plt.tight_layout()

        self.slbInteractx = slbInteractx
        self.slbInteracty = slbInteracty
        self.xparam = xparam
        self.yparam = yparam

        self.xspace = None
        self.yspace = None

        self.plotquantity = plotquantity
        self.unit = unit
        self.normalize = normalizer is not None
        self.normalizer = normalizer

        self.s_exp = s_exp
        if s_exp is not None:
            wexp, Iexpcalib = s_exp.get(plotquantity, Iunit=unit)
        self.wexp = wexp
        self.Iexpcalib = Iexpcalib

#        self.fitroom = None          # type: FitRoom

        self.spectra = {}    # hold the calculated spectra
        self.slabsl = {}    # hold the calculated slabs lists
        self.fconfigs = {}    # hold the calculated slab configs

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def update_markers(self, fconfig, i, j):

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        if self.fitroom.selectTool is not None:
            self.fitroom.selectTool.update_markers(fconfig, i, j)
        else:
            print('... No case selector defined')
        return

    def plot_all_slabs(self, s, slabs):

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        if self.fitroom.slabsTool is not None:
            self.fitroom.slabsTool.spectrum = s
            self.fitroom.slabsTool.slabs = slabs
            self.fitroom.slabsTool.plot_all_slabs(s, slabs)
        else:
            print('log ... No MultiSlabPlot defined')
        return

    def format_coord(self, x, y):
        if self.normalize:
            return 'x = {0:.2f} nm, y = {1:.4f} {2}  '.format(x, y, '(norm)')
        else:
            return 'x = {0:.2f} nm, y = {1:.4f} {2}  '.format(x, y, self.unit)

    def update_slit(self):

        slit_function = self.fitroom.solver.slit
        slit_options = self.fitroom.solver.slit_options
        spectra = self.spectra
        fconfigs = self.fconfigs

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', "interpolating slit function over spectrum grid")
            for (i, j) in spectra.keys():
                spectra[(i, j)].apply_slit(slit_function, verbose=False, **slit_options)
                #print('debug... apply slit for {0} {1}'.format(i,j))
                self.plot_case(i, j, **fconfigs[(i, j)])  # (j,i) not (i,j)
        self.fig.canvas.draw()

    def calc_case(self, i, j, **slabsconfig):
        ''' notice j, i and not i, j 
        i is y, j is x? or the other way round. It's always complicated
        with indexes anyway... (y goes up but j goes down) you see what i mean
        it works, anyway '''

        spectra = self.spectra
        slabsl = self.slabsl
        fconfigs = self.fconfigs

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        if self.fitroom.solver is None:
            raise ValueError('No SlabsConfigSolver defined')

        calc_slabs = self.fitroom.solver.calc_slabs
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', "interpolating slit function over spectrum grid")
            s, slabs, fconfig = calc_slabs(**slabsconfig)
        spectra[(i, j)] = s  # save
        slabsl[(i, j)] = slabs  # save
        fconfigs[(i, j)] = fconfig  # save

    def _plot_failed(self, i, j, error_msg):
        linesim = self.linesim
        legends2 = self.legends2
        try:
            w, ydata = linesim[(i, j)].get_data()
            linesim[(i, j)].set_data(w*nan, ydata*nan)
            legends2[(i, j)].texts[0].set_text(error_msg)
            print(error_msg)
        except KeyError:
            pass   # spectrum not plotted yet. Do nothing
        self.update_markers(None, i, j)
        # If centered, also update the multislab tool
        if i == 1 and j == 1:
            self.plot_all_slabs(None, None)

        return

    def plot_case(self, i, j, **slabsconfig):
        ''' notice j, i and not i, j 
        i is y, j is x? or the other way round. It's always complicated
        with indexes anyway... (y goes up but j goes down) you see what i mean
        it works, anyway '''

        ax2 = self.ax
        lineexp = self.lineexp
        linesim = self.linesim
        legends2 = self.legends2

        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        plotquantity = self.plotquantity
        unit = self.unit
        normalize = self.normalize
        norm_on = self.normalizer

        wexp = self.wexp
        Iexpcalib = self.Iexpcalib

        get_residual = self.fitroom.solver.get_residual

        axij = ax2[j][i]   # note it's (j,i) not (i,j)
        axij.format_coord = self.format_coord

        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        try:
            lineexp[(i, j)]  # does not change anyway .set_data(wexp, ydata)
        except KeyError:
            lines, = plot_stack(wexp, ydata, '-k', lw=2, ax=axij)
            lineexp[(i, j)] = lines

        # Get calculated spectra
        # type: Spectrum # saved by calc_case. None if failed
        s = self.spectra[(i, j)]
        # type: list     # saved by calc_case. None if failed
        slabs = self.slabsl[(i, j)]
        # type: dict     # saved by calc_case. None if failed
        fconfig = self.fconfigs[(i, j)]

        if s is None:
            # TODO : could use 'slabs' as a custom error message
            self._plot_failed(
                i, j, error_msg='Spectrum could not be calculated')
            return

        # calculate residuals
        res = get_residual(s)
        w, I = s.get(plotquantity, wunit='nm', Iunit=unit)

        ydata = norm_on(w, I) if normalize else I

        try:
            linesim[(i, j)].set_data(w, ydata)
            legends2[(i, j)].texts[0].set_text('res: {0:.3g}'.format(res))
        except KeyError:
            line, = axij.plot(w, ydata, 'r')
            linesim[(i, j)] = line
            legends2[(i, j)] = axij.legend((line,), ('res: {0:.3g}'.format(res), ),
                                           loc='upper left', prop={'size': 10})

        self.update_markers(fconfig, i, j)

        # Remember than case (i,j) corresponds to ax2[j,i] which means: j = rows,
        # i = columns
        if j == 2:
            axij.set_xlabel('Wavelength')
        if i == 0:
            if yparam in ['mole_fraction', 'path_length']:  # special format
                axij.set_ylabel('{0} {1:.2g}'.format(
                    yparam, fconfig[slbInteracty][yparam]))
            else:
                axij.set_ylabel('{0} {1:.1f}'.format(
                    yparam, fconfig[slbInteracty][yparam]))
        if j == 0:
            if xparam in ['mole_fraction', 'path_length']:  # special format
                axij.set_title('{0} {1:.2g}'.format(
                    xparam, fconfig[slbInteractx][xparam]), size=20)
            else:
                axij.set_title('{0} {1:.1f}'.format(
                    xparam, fconfig[slbInteractx][xparam]), size=20)
        # TODO: add a set of all labels on line, instead (deals with different values per line)

        # If centered, also update the multislab tool
        if i == 1 and j == 1:
            self.plot_all_slabs(s, slabs)

    def plot_for_export(self, style=['origin'], cases=[(1,1), (0,1), (2,1)],
                        ls='-', xlim=None, ylim=None, labelvar='xy'):
        ''' Sum all center column in one case 
        
        Parameters
        ----------
        
        cases: list
            list of [(row, column)] to plot 
            
        ls: str ('-', '.-', etc.), list, or dict
            if str, use the same. If list, rotate. If dict, use ``cases`` 
            as keys.
            
#            the first one is plot in solid line, the others in alternate with
#            '-.', ':', '-.'
            
        Other Parameters
        ----------------
        
        labelvar: 'x', 'y', 'xy'
            which variable to add. default 'xy'
            Ex::
                
                Tvib=, Trot=
                Tvib=
                Trot=
            
        '''

        set_style(style)

        fig2, ax2 = plt.subplots()
#        ax2 = self.ax

        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        plotquantity = self.plotquantity
        unit = self.unit
        normalize = self.normalize
        norm_on = self.normalizer

        wexp = self.wexp
        Iexpcalib = self.Iexpcalib

        axij = ax2
        axij.format_coord = self.format_coord

        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        plot_stack(wexp, ydata, '-', lw=2, ax=axij, label='Experiment')
        
        for index, (j, i) in enumerate(cases):   # reversed? seems more logical this way.
#            color = ['b', 'r', 'g'][(i+j)]

            # Get calculated spectra
            # type: Spectrum # saved by calc_case. None if failed
            s = self.spectra[(i, j)]
            # type: dict     # saved by calc_case. None if failed
            fconfig = self.fconfigs[(i, j)]

            if s is None:
                # TODO : could use 'slabs' as a custom error message
                self._plot_failed(
                    i, j, error_msg='Spectrum could not be calculated')
                return

            # calculate residuals
            w, I = s.get(plotquantity, wunit='nm', Iunit=unit)

            ydata = norm_on(w, I) if normalize else I
            
            xvalue = fconfig[slbInteractx][xparam]
            yvalue = fconfig[slbInteracty][yparam]
            
            # Label of each plot
            if labelvar == 'xy':
                label = make_up('{0} {1:.0f}K {2} {3:.0f}K'.format(xparam,xvalue,
                                                                   yparam,yvalue))
            elif labelvar == 'x':
                label = make_up('{0} {1:.0f}K'.format(xparam,xvalue))
            elif labelvar == 'y':
                label = make_up('{0} {1:.0f}K'.format(yparam,yvalue))
                
            # Style:
            if isinstance(ls, list):
                ls_i = ls[index%len(ls)]
#            if index == 0:  # first one
#                ls = '-'
#            elif index % 3 == 0:
#                ls = '-.'
#            elif index % 3 == 1:
#                ls = ':'
#            else:
#                ls = '--'
            elif isinstance(ls, dict):
                ls_i = ls[(j, i)]
            else:
                ls_i = ls
            
            axij.plot(w, ydata, label=label, ls=ls_i)

            self.update_markers(fconfig, i, j)

            # Remember than case (i,j) corresponds to ax2[j,i] which means: j = rows,
            # i = columns
#            if j == 2:
#            if i == 0:
#                if yparam in  ['mole_fraction', 'path_length']: # special format
#                    axij.set_ylabel('{0} {1:.2g}'.format(yparam, fconfig[slbInteracty][yparam]))
#                else:
#                    axij.set_ylabel('{0} {1:.1f}'.format(yparam, fconfig[slbInteracty][yparam]))
#            if j == 0:
#                if xparam in  ['mole_fraction', 'path_length']: # special format
#                    axij.set_title('{0} {1:.2g}'.format(xparam, fconfig[slbInteractx][xparam]), size=20)
#                else:
#                    axij.set_title('{0} {1:.1f}'.format(xparam, fconfig[slbInteractx][xparam]), size=20)
            # TODO: add a set of all labels on line, instead (deals with different values per line)

        axij.set_xlabel('Wavelength (nm)')   # hardcoded. May need to change that
        if self.normalize:
            axij.set_ylabel('I (norm)')
        else:
            axij.set_ylabel('I ({0})'.format(self.unit))

        fix_style(style, ax=axij)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.legend(loc='best', fontsize=18)        
        return fig2, ax2

    def _add_multicursor(self):
        ''' Add vertical bar (if not there already)'''

        if self.multi2 is None:
            ax = self.ax
#            multi2 = MultiCursor(self.fig.canvas, (*ax[0], *ax[1], *ax[2]),
#                                 color='r', lw=1,
#                                alpha=0.2, horizOn=False, vertOn=True)
           # Python 2 compatible (but ugly... switch to Python3 now!)
            multi2 = MultiCursor(self.fig.canvas, (ax[0][0], ax[0][1], ax[0][2],
                                                   ax[1][0], ax[1][1], ax[1][2],
                                                   ax[2][0], ax[2][1], ax[2][2]),
                                 color='r', lw=1,
                                 alpha=0.2, horizOn=False, vertOn=True)
            self.multi2 = multi2
        else:
            pass

    def plot_3times3(self, xspace=None, yspace=None):

        if xspace is None:
            xspace = self.xspace  # use last ones
        if yspace is None:
            yspace = self.yspace  # use last ones

        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        # update defaults
        self.xspace = xspace
        self.yspace = yspace

        # update center
        self.fitroom.Slablist[slbInteractx][xparam] = xspace[1]
        self.fitroom.Slablist[slbInteracty][yparam] = yspace[1]

        fig2 = self.fig

        # dont calculate these when the figure is not shown (for performance)
        if self.fitroom.perfmode:
            try:  # works in Qt
                updateSideAxes = not fig2.canvas.manager.window.isMinimized()
            except:
                updateSideAxes = True
        else:
            updateSideAxes = True

        # Do the calculations
        for i, xvari in enumerate(xspace):
            for j, yvarj in enumerate(yspace[::-1]):
                if not (i == 1 and j == 1) and not updateSideAxes:
                    continue
                config0 = self.fitroom.get_config()   # create a copy
                config0[slbInteractx][xparam] = xvari
                config0[slbInteracty][yparam] = yvarj
                self.fitroom.eval_dynvar(config0)  # update dynamic parameters
                self.calc_case(i, j, **config0)   # here we calculate
                self.plot_case(i, j, **config0)   # here we plot

        # Plot title with all slabs conditions
        # use last one, dont print variable parameter
        del config0[slbInteractx][xparam]
        # note that DynPara are the last one only!
        del config0[slbInteracty][yparam]
        # TODO: use intersect dict function
        msg = ''
        for k, cfgi in config0.items():
            msg += k+' - '
            msg += ' '.join(
                ['{0}:{1:.3g}'.format(k, float(v)) for (k, v) in cfgi.items()
                 if not k in ['db', 'factory', 'bandlist', 'source',
                              'overpopulation',  # readable but too many lines
                              ]])
            msg += ' || '
        msg = msg[:-4]
        msg = textwrap.wrap(msg, width=200)
        fig2.suptitle('\n'.join(msg), size=10)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.93-0.02*len(msg))

        # Add cursor than spans over all subplots
        self._add_multicursor()

        # Show figure
        fig2.canvas.show()
        plt.show()
        plt.pause(0.05)
