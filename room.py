# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 01:58:04 2017

@author: erwan
"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from neq.misc.debug import printdbg

try:
    from neq.math.fitroom import CaseSelector
    from neq.math.fitroom import Grid3x3
    from neq.math.fitroom import MultiSlabPlot
    from neq.math.fitroom import SlabsConfigSolver
    from neq.math.fitroom import Overpopulator
    from neq.math.fitroom import SlitTool
except:
    from .selection_tool import CaseSelector
    from .grid3x3_tool import Grid3x3
    from .multislab_tool import MultiSlabPlot
    from .solver import SlabsConfigSolver
    from .noneq_tool import Overpopulator
    from .slit_tool import SlitTool
    
class FitRoom():
    
    def __init__(self, Slablist, perfmode=False):
        ''' 
        Input
        -------
        
        perfmode: boolean
            if True we try to optimize calculation times (ex: minimized windows
            are not recalculated)
        '''
        self.tools = []
        
        # all possible tools:
        self.solver = None      # type: Solver
        self.gridTool = None    # type: Grid3x3
        self.slabsTool = None   # type: MultiSlabPlot
        self.selectTool = None  # type: CaseSelector
        self.overpTool = None   # type: Overpopulator
        self.slitTool = None    # type: SlitTool
        
        self.Slablist = Slablist
        self.perfmode = perfmode
    
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
        elif isinstance(tool, SlitTool):
            print('Adding SlitTool')
            self.slitTool = tool
        else:
            raise ValueError('Unknown tool: {0} ({1})'.format(tool, type(tool)))
        
        # Update links:
        self.tools.append(tool)
        tool.connect(self)
    
    
    def update_plots(self):
    
        perfmode = self.perfmode
    
        # Update GridTool
        if self.gridTool is not None:
            fig2 = self.gridTool.fig
            try:  # works in Qt
                updatefig = not fig2.canvas.manager.window.isMinimized()
            except:
                updatefig = True
            if updatefig or not perfmode:
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
            if updatefig or not perfmode:
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
        if self.slitTool is not None:
            self.slitTool.update_figure()  # in case
            
    def get_config(self):
        ''' Get values for Target configuration '''
        
        Slablist = self.Slablist
        config0 = {k:c.copy() for k, c in Slablist.items()}

        return config0
    
    
    def eval_dynvar(self, config):
        ''' Evaluate dynamic links for a given configuration. Changes updated
        config (inplace) '''
        
        # Evaluate dynamic quantities
        for slabname, slab in config.items():
            for k, v in slab.items():
                if isinstance(v, DynVar):
                    val = v.eval(config)
                    config[slabname][k] = val
                    if __debug__: printdbg("{0}['{1}'] evaluated as {2}".format(
                            slabname, k, val))
                    
        

class DynVar():
    ''' To allow dynamic (runtime) filling of conditions in SlabList
    
    Ex:
            
    > slbPostCO2 = {
    >          'db':db0,
    >          'Tgas':500,
    >          'db':dbp,
    >          'Tgas':1100,
    >          'path_length':0.7,
    >          'mole_fraction':0.35,
    >          }
    > 
    > slbPostCO = {
    >          'db':dbco,
    >          'Tgas':slbPostCO2['Tgas'],
    >          'path_length':DynVar('sPostCO2', 'path_length', lambda x:x),  
    >                        # evaluated at runtime using names in Slablist
    >          'mole_fraction':0.01,
    >          }
    > 
    > Slablist = {
    >             'sPostCO2': slbPostCO2,
    >             'sPostCO': slbPostCO,
    >             }
    
    '''
    
    def __init__(self, slab, param, func=lambda x:x):
        ''' 
        Input 
        ------
        
        slab: str
            slab config name in Slablist
            
        param: str
            param name in slab config dict
            
        func: function
            function to apply. Default identity
        '''
        
        self.slab = slab
        self.param = param
        self.func = func
        
        return
    
    def eval(self, slabsconfig):
        ''' Evaluate value at runtime based on other static values '''
        
        if isinstance(slabsconfig[self.slab][self.param], DynVar):
            raise AssertionError("A DynVar cannot refer to another DynVar {0}['{1}']".format(
                    self.slab, self.param))
        
        return self.func(slabsconfig[self.slab][self.param])
    
#    ('sPlasmaCO2', 'mole_fraction')
