# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 01:58:04 2017

@author: erwan
"""

import matplotlib.pyplot as plt

try:
    from neq.math.fitroom import CaseSelector
    from neq.math.fitroom import Grid3x3
    from neq.math.fitroom import MultiSlabPlot
    from neq.math.fitroom import SlabsConfigSolver
    from neq.math.fitroom import Overpopulator
except:
    from .selection_tool import CaseSelector
    from .grid3x3_tool import Grid3x3
    from .multislab_tool import MultiSlabPlot
    from .solver import SlabsConfigSolver
    from .noneq_tool import Overpopulator

class FitRoom():
    
    def __init__(self, Slablist):
        self.tools = []
        self.solver = None
        self.gridTool = None
        self.slabsTool = None
        self.selectTool = None
        self.overpTool = None
        
        self.Slablist = Slablist
    
    def add_tool(self, tool):
        if isinstance(tool, SlabsConfigSolver):
            print('Adding SlabsConfigSolver')
            self.solver = tool
        elif isinstance(tool, Grid3x3):
            print('Adding Grid3x3')
            self.gridTool = tool
        elif isinstance(tool, MultiSlabPlot):
            print('Adding MultiSlabPlot')
            self.slabsTool = tool
        elif isinstance(tool, CaseSelector):
            print('Adding CaseSelector')
            self.selectTool = tool
        elif isinstance(tool, Overpopulator):
            print('Adding Overpopulator')
            self.overpTool = tool
        
        # Update links:
        self.tools.append(tool)
        tool.fitroom = self
    
    
    def update_plots(self):
    
    
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
                
            
    def update(self, xspace=None, yspace=None):
            
        if self.gridTool is not None:
            # Update gridTool (updating slabsTool is done in the middle of the 
            # loop too)
            self.gridTool.plot_3times3(xspace, yspace)
        elif self.slabsTool is not None:
            self.slabsTool.update()
        else:
            raise ValueError('Neither GridTool or SlabsTool defined')
            
            
            
    def get_config(self):
        ''' Get values for Target configuration '''
        
        Slablist = self.Slablist
        config0 = {k:c.copy() for k, c in Slablist.items()}

        return config0
        
