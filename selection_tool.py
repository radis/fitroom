# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:41:43 2017

@author: erwan

Description
-----

A window to select conditions along two axis (to calculate, or retrieve
from a database)

Todo
-----

interface
 - used keyboards keys to move rectangle selector

"""

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from numpy import array, meshgrid, empty_like, linspace
from scipy.interpolate import griddata

class CaseSelector():

    def __init__(self, dbInteractx=None, dbInteracty=None, xparam='', yparam='',
                 slbInteractx=None, slbInteracty=None, nfig=None,
                 solver=None, gridTool=None, slabsTool=None,
                 xmin=0, xmax=0, ymin=0, ymax=0):


        # Init variables        
        self.linemarkers = {}

        self.solver = solver
        self.gridTool = gridTool
        self.slabsTool = slabsTool

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
                                           button=[1], #, 3],  # left button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
        plt.connect('key_press_event', self.toggle_selector)

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


    def _plot_db_params(self): 
        ''' Plot database '''
        
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
        
        x = dbInteractx.df[xparam]
        y = dbInteracty.df[yparam]
    
        # Plot
        fig, ax = plt.subplots(num=nfig)
        if dbInteractx == dbInteracty:
            ax.plot(x, y, 'ok')
            
        else:
            xx, yy = meshgrid(list(set(x)), list(set(y)))
            ax.plot(xx, yy, 'ok')
        # TODO: add units from spectrum here. (but maybe units arent the same for all database????)
        # load it up first and check? 
        ax.set_xlabel('{0} {1}'.format(slbInteractx, xparam))
        ax.set_ylabel('{0} {1}'.format(slbInteracty, yparam))
        
        return fig, ax

    def update_action(self, xmin, xmax, ymin, ymax):
        if self.gridTool is not None:
            
            xcen = (xmin + xmax)/2
            ycen = (ymin + ymax)/2
            
            self.gridTool.plot_3times3([xmin, xcen, xmax], [ymin, ycen, ymax])
        else:
            print('log ... No gridTool defined')
        return

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'

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
            
            # Update GridTool
            if self.gridTool is not None:
                fig2 = self.gridTool.fig
                try:  # works in Qt
                    updatefig = not fig2.canvas.manager.window.isMinimized()
                except:
                    updatefig = True
                if updatefig:
                    plt.figure(2).show()
                    plt.pause(0.1)  # make sure the figure is replotted
#            else:
#                print('Log: no gridtool')
                
            # Update SlabsTool
            if self.slabsTool is not None:
                fig3 = self.slabsTool.fig
                try:  # works in Qt
                    updatefig = not fig3.canvas.manager.window.isMinimized()
                except:
                    updatefig = True
                if updatefig:
                    plt.figure(3).show()
                    plt.pause(0.1)  # make sure the figure is replotted
                    
#            else:
#                print('Log: no slabstool')
                    
                
                
            self.RS.set_active(True)
            plt.ion()

        except:
            import sys
            print(sys.exc_info())
            raise

    def toggle_selector(self, event):
        if event.key in ['Q', 'q'] and self.RS.active:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)

    def update_markers(self, i, j, *markerpos):
        
        x, y= markerpos
        try:
            self.linemarkers[(i,j)].set_visible(True)
            self.linemarkers[(i,j)].set_data(x, y)
        except KeyError:
            line, = self.ax.plot(x, y, 'or', markersize=12, mfc='none')
            self.linemarkers[(i,j)] = line
        return
            
    
    def precompute_residual(self, Slablist):
        ''' Plot residual for all points in database '''
    
        if self.solver is None:
            raise ValueError('No solver defined')
        
        dbInteractx = self.dbInteractx
        dbInteracty = self.dbInteracty
        slbInteractx = self.slbInteractx
        slbInteracty = self.slbInteracty
        xparam = self.xparam
        yparam = self.yparam
        
        calc_slabs = self.solver.calc_slabs
        get_residual = self.solver.get_residual
        
        ax1 = self.ax
        fig1 = self.fig 
        
        if dbInteractx == dbInteracty and False:
            ''' Doesnt work... fix later?
            I think it doesnt like the sorting
            '''
    
    
            # only calculate database points
            xspace, yspace = zip(*array(dbInteractx.view()[[xparam, yparam]]))
            # kill duplicates
            xspace, yspace = zip(*set(list(zip(xspace, yspace))))
    
            xx, yy = meshgrid(xspace, yspace)
    
            res = []  #np.empty_like(xx)
    
            for xvari, yvarj in zip(xspace, yspace):
                config0 = {k:c.copy() for k, c in Slablist.items()}
                config0[slbInteractx][xparam] = xvari
                config0[slbInteracty][yparam] = yvarj
    
            #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"
    
                s, slabs, fconfig = calc_slabs(**config0)
    
                resij = get_residual(s)
    
                res.append(resij)
    
            res = array(res)
            # Create a 2D grid by interpolating database data
            res = griddata((xspace, yspace), res, (xx, yy))
    
        else:
            # do a mapping of all possible cases
            xspace = array(sorted(set(dbInteractx.view()[xparam])))
            yspace = array(sorted(set(dbInteracty.view()[yparam])))
    
            xx, yy = meshgrid(xspace, yspace, indexing='ij')
    
            res = empty_like(yy)
    
            for i, xvari in enumerate(xspace):
                for j, yvarj in enumerate(yspace):
                    config0 = {k:c.copy() for k, c in Slablist.items()}
                    config0[slbInteractx][xparam] = xvari
                    config0[slbInteracty][yparam] = yvarj
    
                #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"
    
                    s, slabs, fconfig = calc_slabs(**config0)
    
                    resij = get_residual(s)
    
                    res[i][j] = resij
    
        cf = ax1.contourf(xx, yy, res, 40, cmap=plt.get_cmap('viridis_r'))
        cbar = fig1.colorbar(cf)
        cbar.ax.set_ylabel('residual')
        plt.tight_layout()
    
    
