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

class CaseSelector():

    def __init__(self, ax1, fig2, fig3, gridTool=None):

#        self.fig = plt.figure()
        self.ax1 = ax1
        self.fig2 = fig2
        self.fig3 = fig3
        
        self.linemarkers = {}

        self.gridTool = gridTool

        self.RS = RectangleSelector(ax1, self.line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1], #, 3],  # left button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
        plt.connect('key_press_event', self.toggle_selector)

    def update_action(self, xmin, xmax, ymin, ymax):
        if self.gridTool is not None:
            
            xcen = (xmin + xmax)/2
            ycen = (ymin + ymax)/2
            
            self.gridTool.plot_3times3([ymin, ycen, ymax], [xmin, xcen, xmax])  # yes i flipped it -_-
        else:
            print('No gridTool defined')
        return

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'

        fig3 = self.fig3
        fig2 = self.fig2

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
            try:  # works in Qt
                updatefig = not fig2.canvas.manager.window.isMinimized()
            except:
                updatefig = True
            if updatefig:
                plt.figure(2).show()
                plt.pause(0.1)  # make sure the figure is replotted
            try:  # works in Qt
                updatefig = not fig3.canvas.manager.window.isMinimized()
            except:
                updatefig = True
            if updatefig:
                plt.figure(3).show()
                plt.pause(0.1)  # make sure the figure is replotted
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
        
        try:
            self.linemarkers[(i,j)].set_visible(True)
            self.linemarkers[(i,j)].set_data(*markerpos)
        except KeyError:
            line, = self.ax1.plot(*markerpos, 'or', markersize=12, mfc='none')
            self.linemarkers[(i,j)] = line
        return
            