# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:46:31 2017

@author: erwan

Non-equilibrium tool to define overpopulations 
"""


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from neq.phys.conv import cm2eV


class Overpopulator():

    def __init__(self, slab, levels='all', nfig=None):
        ''' 
        Parameters
        ----------

        slab:
            slab to connect to. Must be a slab calculated with from_band source 
            mode

        '''

        self.nfig = nfig

        if nfig is None:
            nfig = 'Overpopulation'

        # Init figure
        fig = plt.figure(nfig)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_ylim((0, 3))

        self.fig = fig
        self.ax = ax
        self.slab = slab
        lvlist = slab['bandlist']

        # Get levels to adjust. If 'all' get all levels in lvlist
        if levels == 'all':
            levels = list(lvlist.vib_levels.index)

        self.levels = levels
        self.lvlist = lvlist
        E_lvls = lvlist.vib_levels.loc[levels, 'E']
        self.E_lvls = E_lvls

        # initial overpopulation distribution
        if 'overpopulation' in slab:
            overpopulation = slab['overpopulation']
        else:
            overpopulation = {}

        circles = {}
        for i, lvl in enumerate(levels):
            xcoord = float(E_lvls.loc[lvl])
#            xcoord = cm2eV(E_lvls.loc[lvl])

            try:
                ycoord = overpopulation[lvl]
            except KeyError:
                ycoord = 1

            print((lvl, xcoord, 'cm-1'))
#            circles[lvl] = patches.Ellipse((xcoord, ycoord), 0.03, 4.5*0.03, fc='k', alpha=0.7)
            circles[lvl] = patches.Ellipse(
                (xcoord, ycoord), 100, 4.5*0.03, fc='k', alpha=0.7)

        self.circles = circles

        drs = []
        for br, circ in circles.items():
            ax.add_patch(circ)
            dr = DraggablePoint(circ, self.action)
            dr.connect()
            drs.append(dr)
        self.drs = drs

        plt.show()
        plt.xlim((0, E_lvls.max()))
        plt.xlabel('Energy (cm-1)')
        plt.ylabel('Overpopulation')
        plt.tight_layout()

        return

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def action(self):
        self.update_overpopulation()
        self.fitroom.update()
        self.fitroom.update_plots()

    def update_overpopulation(self):
        self.slab['overpopulation'] = self.get()

    def get(self):

        overpopulation = {}
        for br in self.circles:
            circ = self.circles[br]
            overpopulation[br] = circ.center[1]

        return overpopulation


class DraggablePoint:
    ''' A draggable point. Modified to allow vertical movement only 


    Initial source code
    ---------

    from Idealist 
    @https://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    '''
    lock = None  # only one can be animated at a time

    def __init__(self, point, action=None):
        self.point = point
        self.press = None
        self.background = None
        self.action = action

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes:
            return
        if DraggablePoint.lock is not None:
            return
        contains, attrd = self.point.contains(event)
        if not contains:
            return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes:
            return
        self.point.center, xpress, ypress = self.press
#        dx = event.xdata - xpress
        dx = 0
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # Do something
        if self.action is not None:
            self.action()

        # redraw the full figure
        self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
