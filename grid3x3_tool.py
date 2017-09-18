# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:22:30 2017

@author: erwan

Tool to plot on 9 graphes along 2 axes (2 conditions)
"""

import matplotlib.pyplot as plt
import textwrap
from matplotlib.widgets import MultiCursor

class Grid3x3():

    def __init__(self, slbInteractx=None, slbInteracty=None,
                 xparam='', yparam='',
                 plotquantity='radiance', unit= 'mW/cm2/sr/nm',
                 normalizer=None,
                 wexp=None, Iexpcalib=None, wexp_shift=0,
                 # Other params
                 Slablist = None,
                 ):

        plt.figure(2, figsize=(16, 12)).clear()
        fig2, ax2 = plt.subplots(3, 3, sharex=True, sharey=True,
                               num=2)

        self.fig = fig2
        self.ax = ax2
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

        self.wexp = wexp
        self.Iexpcalib = Iexpcalib
        self.wexp_shift = wexp_shift

        self.fitroom = None
#        self.SlabsConfigSolver = SlabsConfigSolver
#        self.CaseSelector = CaseSelector
#        self.MultiSlabPlot = MultiSlabPlot

        self.Slablist = Slablist

    def update_markers(self, i, j, *markerpos):

        if self.fitroom is None:
            raise ValueError('Tool not connected to Fitroom')
        if self.fitroom.selectTool is not None:
            self.fitroom.selectTool.update_markers(i, j, *markerpos)
        else:
            print('... No case selector defined')
        return

    def plot_all_slabs(self, s, slabs):
        
        if self.fitroom is None:
            raise ValueError('Tool not connected to Fitroom')
        if self.fitroom.slabsTool is not None:
            self.fitroom.slabsTool.plot_all_slabs(s, slabs)
        else:
            print('log ... No MultiSlabPlot defined')
        return

    def format_coord(self, x, y):
        return 'x = {0:.2f} nm, y = {1:.4f} {2}  '.format(x, y, self.unit)

    def plot_case(self, j, i, **slabsconfig):
        ''' notice j, i and not i, j 
        i is y, j is x? or the other way round. It's always complicated
        with indexes anyway... (y goes up but j goes down) you see what i mean
        it works with this anyway '''

        if self.fitroom is None:
            raise ValueError('Tool not connected to Fitroom')
        if self.fitroom.solver is None:
            raise ValueError('No SlabsConfigSolver defined')
            
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
        wexp_shift = self.wexp_shift

        calc_slabs = self.fitroom.solver.calc_slabs
        get_residual = self.fitroom.solver.get_residual

        axij = ax2[i][j]
        axij.format_coord = self.format_coord

        ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
        try:
            lineexp[(i,j)].set_data(wexp+wexp_shift, ydata)
        except KeyError:
            line, = axij.plot(wexp+wexp_shift, ydata,'-k',lw=2)
            lineexp[(i,j)] = line

        s, slabs, fconfig = calc_slabs(**slabsconfig)

        res = get_residual(s)

        w, I = s.get(plotquantity, xunit='nm', yunit=unit)
        
        # Get final values
        xmarker = fconfig[slbInteractx][xparam]
        ymarker = fconfig[slbInteracty][yparam]
        markerpos = (xmarker, ymarker)

        ydata = norm_on(w, I) if normalize else I
        
        try:
            linesim[(i,j)].set_data(w, ydata)
            legends2[(i,j)].texts[0].set_text('res: {0:.3g}'.format(res))
        except KeyError:
            line, = axij.plot(w, ydata, 'r')
            linesim[(i,j)] = line
            legends2[(i,j)] = axij.legend((line,), ('res: {0:.3g}'.format(res), ),
                    loc='upper left', prop={'size':10})
        
        self.update_markers(i, j, *markerpos)
        
        if i == 2: axij.set_xlabel('Wavelength')
        if j == 0:
            if yparam == 'mole_fraction':
                axij.set_ylabel('{0} {1:.2g}'.format(yparam, fconfig[slbInteracty][yparam]))
            else:
                axij.set_ylabel('{0} {1:.1f}'.format(yparam, fconfig[slbInteracty][yparam]))
        if i == 0:
            if xparam == 'mole_fraction':
                axij.set_title('{0} {1:.2g}'.format(xparam, fconfig[slbInteractx][xparam]), size=20)
            else:
                axij.set_title('{0} {1:.1f}'.format(xparam, fconfig[slbInteractx][xparam]), size=20)
        #TODO: add a set of all labels on line, instead (deals with different values per line)

        if i == 1 and j == 1:
            self.plot_all_slabs(s, slabs)


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
            xspace = self.xspace # use last ones
        if yspace is None:
            yspace = self.yspace # use last ones
        
        # update defaults
        self.xspace = xspace
        self.yspace = yspace

        Slablist = self.Slablist
        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        fig2 = self.fig

        config0 = {k:c.copy() for k, c in Slablist.items()}

        # dont calculate these when the figure is not shown (for performance)
        try:  # works in Qt
            updateSideAxes = not fig2.canvas.manager.window.isMinimized()
        except:
            updateSideAxes = True

        for i, xvari in enumerate(xspace):
            for j, yvarj in enumerate(yspace[::-1]):
                if not (i==1 and j==1) and not updateSideAxes: continue
                config0[slbInteractx][xparam] = xvari
                config0[slbInteracty][yparam] = yvarj
                self.plot_case(i, j, **config0)

        # Plot title with all slabs conditions
        del config0[slbInteractx][xparam]      # dont print variable parameter
        del config0[slbInteracty][yparam]
        msg = ''
        for k, cfgi in config0.items():
            msg += k+' - '
            msg += ' '.join(
                ['{0}:{1:.3g}'.format(k,v) for (k,v) in cfgi.items() 
                if not k in ['db','factory', 'bandlist','source',
                             'overpopulation',  # readable but too many lines
                             ]])
            msg += ' || '
        msg = msg[:-4]
        msg = textwrap.wrap(msg, width=200)
        fig2.suptitle('\n'.join(msg), size=10)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.93-0.02*len(msg))

        self._add_multicursor()

    #    plt.figure(2).canvas.show()
        fig2.canvas.show()
        plt.show()
        plt.pause(0.05)
