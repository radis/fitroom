# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:41:43 2017

@author: erwan

Summary
-------

A window to select conditions along two axis (to calculate, or retrieve
from a database)

Notes
-----

# TODO: interface
 - used keyboards keys to move rectangle selector

-------------------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from numpy import array, meshgrid, empty_like, linspace
from scipy.interpolate import griddata
from six.moves import zip
import sys
from radis.misc.progress_bar import ProgressBar
#from neq.math.fitroom.solver import expand_columns


class CaseSelector():
    '''

    Todo
    ---------

    prevent using DynVar as xparam, yparam

    '''

    def __init__(self, dbInteractx=None, dbInteracty=None, xparam='', yparam='',
                 slbInteractx=None, slbInteracty=None, nfig=None,
                 xmin=0, xmax=0, ymin=0, ymax=0,
                 plot_data_color='k'):
        ''' Main tool to choose which cases to plot 
        
        Examples
        --------
        
        See the working case in :mod:`~neq.test.math.test_fitroom`. In particular, run
        :func:`~neq.test.math.test_fitroom.test_start_fitroom`
            
        See Also
        --------
        
        :class:`~neq.math.fitroom.grid3x3_tool.Grid3x3`,
        :class:`~neq.math.fitroom.multislab_tool.MultiSlabPlot`,
        :class:`~neq.math.fitroom.solver.SlabsConfigSolver`,
        :class:`~neq.math.fitroom.noneq_tool.Overpopulator`,
        :class:`~neq.math.fitroom.room.FitRoom`,
        :class:`~neq.math.fitroom.slit_tool.SlitTool` 
        
        '''

        # Init variables
        self.linemarkers = {}

        self.fitroom = None               # type: FitRoom
#        self.solver = solver
#        self.gridTool = gridTool
#        self.slabsTool = slabsTool

        self.dbInteractx = dbInteractx
        self.dbInteracty = dbInteracty
#        self.factoryx = factoryx
#        self.factoryy = factoryy
        self.slbInteractx = slbInteractx
        self.slbInteracty = slbInteracty
        self.xparam = xparam
        self.yparam = yparam
        self.nfig = nfig

        if dbInteractx is not None and dbInteracty is not None:
            self.mode = 'database'
        else:
            self.mode = 'calculate'
#            assert(factoryx is not None and factoryy is not None)
#            assert(x0=0, y0=0, xstep=0, ystep=0)

        self.plot_data_color=plot_data_color
        
        # Init figure
        if self.mode == 'database':
            fig, ax = self._plot_db_params()
        else:
            fig, ax = self._plot_calc_range(xmin, xmax, ymin, ymax)

        self.ax = ax
        self.fig = fig

        # Add Rectangle Selector to figure
        self.RS = RectangleSelector(ax, self.line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1],  # , 3],  # left button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        plt.connect('key_press_event', self.toggle_selector)

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def _plot_calc_range(self, xmin, xmax, ymin, ymax):

        # Get inputs
        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam
        nfig = self.nfig

        # Plot
        plt.figure(nfig).clear()

        # Plot
        fig, ax = plt.subplots(num=nfig)

        # TODO: add units from spectrum here. (but maybe units arent the same for all database????)
        # load it up first and check?
        ax.set_xlabel('{0} {1}'.format(slbInteractx, xparam))
        ax.set_ylabel('{0} {1}'.format(slbInteracty, yparam))

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

        return fig, ax

    def _plot_db_params(self, **kwargs):
        ''' Plot database 
        
        Parameters
        ----------
        
        kwargs: dict
            forwarded to plot
        '''
        
        # default:
        kwargs.update({'color':self.plot_data_color})

        # Get inputs
        dbInteractx = self.dbInteractx
        dbInteracty = self.dbInteracty
        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam
        nfig = self.nfig

        # Plot
        plt.figure(nfig).clear()

        # Check input
        if len(dbInteractx.df) == 0:
            raise ValueError('Database {0} is empty'.format('dbInteractx'))
        if len(dbInteracty.df) == 0:
            raise ValueError('Database {0} is empty'.format('dbInteracty'))

        x = dbInteractx.df[xparam]
        y = dbInteracty.df[yparam]
 
#        try:
#            x = dbInteractx.df[xparam]
#        except KeyError:
#            # maybe key needs to be expanded. (ex: asking for Tvib1=... while Tvib=(...,...) is given)
#            # note @dev: done as a a posteriori hack/fix for multi Tvib modes. 
#            if xparam[:-1] in dbInteractx.df:
#                dbInteractx.df = expand_columns(dbInteractx.df, [xparam[:-1]])
#                x = dbInteractx.df[xparam]
#            else:
#                raise
#            
#        try:
#            y = dbInteracty.df[yparam]
#        except KeyError:
#            # maybe key needs to be expanded. (ex: asking for Tvib1=... while Tvib=(...,...) is given)
#            # note @dev: done as a a posteriori hack/fix for multi Tvib modes. 
#            if yparam[:-1] in dbInteracty.df:
#                dbInteracty.df = expand_columns(dbInteracty.df, [yparam[:-1]])
#                y = dbInteracty.df[yparam]
#            else:
#                raise
            
        # Plot
        fig, ax = plt.subplots(num=nfig)
        if dbInteractx == dbInteracty:
            ax.plot(x, y, 'o', ms=3, **kwargs)

        else:
            xx, yy = meshgrid(list(set(x)), list(set(y)))
            ax.plot(xx, yy, 'o', ms=3, **kwargs)
        # TODO: add units from spectrum here. (but maybe units arent the same for all database????)
        # load it up first and check?
        ax.set_xlabel('{0} {1}'.format(slbInteractx, xparam))
        ax.set_ylabel('{0} {1}'.format(slbInteracty, yparam))

        return fig, ax

    def update_action(self, xmin, xmax, ymin, ymax):

        xcen = (xmin + xmax)/2
        ycen = (ymin + ymax)/2

        self.update_target_config(xcen, ycen)

        self.fitroom.update([xmin, xcen, xmax], [ymin, ycen, ymax])

        return

    def update_target_config(self, xcen, ycen):

        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        # update center
        Slablist = self.fitroom.Slablist
        Slablist[slbInteractx][xparam] = xcen
        Slablist[slbInteracty][yparam] = ycen

        return

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')

#        if self.slabsTool is None:
#            print('No slabsTool defined. Aborting')
#            return

#        if self.gridTool is None:
#            print('No gridTool defined. Aborting')
#            return

        try:
            plt.ioff()
            self.RS.set_active(False)

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            self.update_action(xmin, xmax, ymin, ymax)

            # This is when the plots are updated:
            # note: to increase perfs if windows is minimized we dont update it
            # this mean it wont be updated once it is visible again, though.

            self.fitroom.update_plots()

            self.RS.set_active(True)
            plt.ion()

        except:
            print('An error occured during selectTool callback')
            import traceback
            traceback.print_exc()
            raise

    def toggle_selector(self, event):
        if event.key in ['Q', 'q'] and self.RS.active:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)

    def update_markers(self, fconfig, i=1, j=1):
        ''' 
        Parameters
        ----------

        i,j : int
            marker position  (different from 1,1 in 3x3 grid )
        '''

        if fconfig is None:
            try:
                self.linemarkers[(i, j)].set_visible(False)
            except KeyError:
                pass
            return

        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        x = fconfig[slbInteractx][xparam]
        y = fconfig[slbInteracty][yparam]

        try:
            self.linemarkers[(i, j)].set_visible(True)
            self.linemarkers[(i, j)].set_data(x, y)
        except KeyError:
            line, = self.ax.plot(x, y, 'or', markersize=12, mfc='none')
            self.linemarkers[(i, j)] = line
        return

    def precompute_residual(self, Slablist, xspace='database', yspace='database',
                            contour='contourf', normalize=False, normalize_how='max',
                            vmin=None, vmax=None):
        ''' Plot residual for all points in database.

        Parameters
        ----------

        Slablist: configuration 

        xspace: array, or 'database'
            values of points to precompute. If 'database', residual is calculated
            for all points in database. Default 'database'.

        yspace: array, or 'database'
            values of points to precompute. If 'database', residual is calculated
            for all points in database. Default 'database'.

        Other Parameters
        ----------------
        
        vmin, vmax: float
            used for colorbar

        Examples
        --------
        
        When ``yparam`` is mole_fraction and we want to calculate for many mole 
        fraction conditions from 0 to 1. ::
            
            selectTool.precompute_residual(Slablist, normalize=normalize,
                               yspace=np.linspace(0.1, 1, 10))
        
        '''
        from warnings import catch_warnings, filterwarnings
        from radis.misc.warning import SlitDispersionWarning

        if not hasattr(self, 'fitroom'):
            raise AttributeError('Tool not connected to Fitroom')
        if self.fitroom.solver is None:
            raise ValueError('No solver defined')

        dbInteractx = self.dbInteractx
        dbInteracty = self.dbInteracty
        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam

        calc_slabs = self.fitroom.solver.calc_slabs
        get_residual = self.fitroom.solver.get_residual

        ax1 = self.ax
        fig1 = self.fig

        if dbInteractx == dbInteracty and False:
            ''' Doesnt work... fix later?
            I think it doesnt like the sorting
            '''

            if xspace == 'database' and yspace == 'database':
                # only calculate database points
                xspace, yspace = list(
                    zip(*array(dbInteractx.view()[[xparam, yparam]])))
                # kill duplicates
                xspace, yspace = list(zip(*set(list(zip(xspace, yspace)))))
            elif xspace == 'database':   # not tested
                xspace = array(sorted(set(dbInteractx.view()[xparam])))
            elif yspace == 'database':  # not tested
                yspace = array(sorted(set(dbInteracty.view()[yparam])))
            else:
                pass   # use xspace, yspace values

            xx, yy = meshgrid(xspace, yspace)

            res = []  # np.empty_like(xx)

            pb = ProgressBar(len(xspace))
            for i, (xvari, yvarj) in enumerate(zip(xspace, yspace)):
                pb.update(i)
                config0 = {k: c.copy() for k, c in Slablist.items()}
                config0[slbInteractx][xparam] = xvari
                config0[slbInteracty][yparam] = yvarj
                self.fitroom.eval_dynvar(config0)  # update dynamic parameters

            #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"

                with catch_warnings():
                    filterwarnings('ignore', category=SlitDispersionWarning)
                    filterwarnings('ignore', category=UserWarning) # ignore all for the moment
                    s, slabs, fconfig = calc_slabs(**config0)

                if s is None:   # spectrum not calculated
                    print('Spectrum not calculated. Couldnt calculate residuals: {0}'.format(
                            fconfig))
                    return

                resij = resij = get_residual(s, normalize=normalize, 
                                             normalize_how=normalize_how)
                res.append(resij)
            pb.done()

            res = array(res)
            # Create a 2D grid by interpolating database data
            res = griddata((xspace, yspace), res, (xx, yy))

        else:
            # do a mapping of all possible cases
            if xspace == 'database':
                xspace = array(sorted(set(dbInteractx.view()[xparam])))
            if yspace == 'database':
                yspace = array(sorted(set(dbInteracty.view()[yparam])))
            # else: use xspace, yspace values

            xx, yy = meshgrid(xspace, yspace, indexing='ij')

            res = empty_like(yy, dtype=np.float64)

            pb = ProgressBar(len(xspace))
            for i, xvari in enumerate(xspace):
                pb.update(i)
                for j, yvarj in enumerate(yspace):
                    config0 = {k: c.copy() for k, c in Slablist.items()}
                    config0[slbInteractx][xparam] = xvari
                    config0[slbInteracty][yparam] = yvarj
                    self.fitroom.eval_dynvar(config0)  # update dynamic parameters

                #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"

                    with catch_warnings():
                        filterwarnings('ignore', category=SlitDispersionWarning)
                        filterwarnings('ignore', category=UserWarning) # ignore all for the moment
                        s, slabs, fconfig = calc_slabs(**config0)
    
                    if s is None:   # spectrum not calculated
                        print('Spectrum not calculated. Couldnt calculate residuals'.format(
                            fconfig))
                        return

                    resij = get_residual(s, normalize=normalize, 
                                             normalize_how=normalize_how)

                    res[i][j] = resij
            pb.done()

        try:
            if contour=='contourf':
                cf = ax1.contourf(xx, yy, res, 40, cmap=plt.get_cmap('viridis_r'),
                                  vmin=vmin, vmax=vmax)
            elif contour=='contour':
                cf = ax1.contour(xx, yy, res, 40, cmap=plt.get_cmap('viridis_r'),
                                  vmin=vmin, vmax=vmax)
            elif isinstance(contour, float):
                # Add your own label
                cf = ax1.contourf(xx, yy, res, 40, cmap=plt.get_cmap('viridis_r'),
                                  vmin=vmin, vmax=vmax)
                cs2 = ax1.contour(xx, yy, res, 40, levels=[contour],
                                  vmin=vmin, vmax=vmax)
#                self.cs2 = cs2
#                self.clabel = ax1.clabel(cs2, cs2.levels, inline=True) 
            else:
                raise ValueError('Unexpected: {0}'.format(contour))
            
        except TypeError:
            print(sys.exc_info())
            raise TypeError('An error occured (see details above). This may be due ' +
                            'to an incorrect range of interpolation points. You can set ' +
                            'up the range manually with xspace= and yspace=)')

        cbar = fig1.colorbar(cf)
        cbar.ax.set_ylabel('residual')
        if isinstance(contour, float):
            cbar.ax.plot([0, 1], [contour]*2, 'k') 

        # Add z value in infobar:
        Xflat, Yflat, Zflat = xx.flatten(), yy.flatten(), res.flatten()

        def fmt(x, y):
            # get closest point with known data
            dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
            idx = np.argmin(dist)
            z = Zflat[idx]
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
        ax1.format_coord = fmt

        plt.tight_layout()


if __name__ == '__main__':
    
    from neq.test.math.test_fitroom import test_start_fitroom
    test_start_fitroom()