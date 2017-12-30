# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:41:43 2017

@author: erwan

Description
-----

Functions to build a 2D, multislabs fitting room with a pure Python (Matplotlib)
interface that displays:

    (1) a window to select conditions along two axis (to calculate, or retrieve
    from a database)

    (2) a window to plot 9 spectra corresponding to left, right and center conditions

    (3) a window to decompose the center slab along the different slabs

Todo
-----

interface
 - used keyboards keys to move rectangle selector

"""

from __future__ import absolute_import
from neq.spec.database import SpecDatabase
from neq.spec.slabs import MergeSlabs, SerialSlabs
import numpy as np
from numpy import linspace
from publib import set_style
from neq.misc.testutils import getTestFile, build_test_databases

try:
    from neq.math.fitroom import CaseSelector
    from neq.math.fitroom import Grid3x3
    from neq.math.fitroom import MultiSlabPlot
    from neq.math.fitroom import SlabsConfigSolver
    from neq.math.fitroom import Overpopulator
    from neq.math.fitroom import FitRoom, DynVar
    from neq.math.fitroom import SlitTool
except:
    from .selection_tool import CaseSelector
    from .grid3x3_tool import Grid3x3
    from .multislab_tool import MultiSlabPlot
    from .solver import SlabsConfigSolver
    from .noneq_tool import Overpopulator
    from .room import FitRoom, DynVar
    from .slit_tool import SlitTool
    
from neq.spec import SpectrumFactory
    
if __name__ == '__main__':
    
    set_style('origin')
    
    # %% Generate test spectra database (just for the example)
    # ------------------------
    
    # Add HITRAN-TEST-CO2 to neq.rc (create file if doesnt exist)
    build_test_databases()
    
    sf2 = SpectrumFactory(
                         wavelength_min=4165,
                         wavelength_max=4200,
                         mole_fraction=1,
                         path_length=0.025, 
                         cutoff=1e-25,
                         isotope=[1,2],
                         db_use_cached=True,
                         medium='air')
    sf2.load_databank('HITRAN-CO2-TEST')
    sf2.init_database(getTestFile('HITRAN_CO2_test_spec_database'), autoretrieve='force')
    
    for Tgas in [300, 350, 400, 1200, 1300, 1500, 1700, 2000]:
        sf2.eq_spectrum(Tgas)
        # Note that autoretrieve is set to True by default, so already generated
        # spectra wont be calculated again
    
    
    # %% Load Database
    # -----------------------------------------------------------------------------
    
    db0 = SpecDatabase(getTestFile('HITRAN_CO2_test_spec_database'))   # CO2
    
    # %% `Plot fit
    
    
    
    slbPlasmaCO2 = {
             'db':db0,
#             'bandlist':bandlistCO2,
             'Tgas':1500,
             'path_length':0.02,
             'mole_fraction':1,
#             'source':'from_bands',
             }
    
    slbPostCO2 = {
             'db':db0,
             'Tgas':375,
    #         'db':dbp,
    #         'Tvib':1100,
    #         'Trot':1200,
             'path_length':0.7,
             'mole_fraction':DynVar('sPlasmaCO2', 'mole_fraction'),
             }
    
    slbRoomCO2 = {
             'db':db0,
             'Tgas':300,
             'path_length':373, #+20,
             'mole_fraction':400e-6,
             }
    
    Slablist = {'sPlasmaCO2':slbPlasmaCO2,
                'sPostCO2': slbPostCO2,
                'sRoomCO2': slbRoomCO2}
    
    def config(**slabs):
        ''' args must correspond to slablist. Indexes and order is important '''
    
        return SerialSlabs(slabs['sPlasmaCO2'], slabs['sPostCO2'], slabs['sRoomCO2'])
    
    # Sensibility analysis
    
    slbInteractx = 'sPostCO2'
    xparam = 'Tgas'
    slbInteracty = 'sPlasmaCO2'
    yparam = 'Tgas'
    #slbInteract = slbPostCO
    #xparam = 'Tgas'
    #yparam = 'path_length'
    xstep = 0.2
    ystep = 0.2
    
    slit = (1.63, 2.33) 

    verbose=False
    normalize = False
    precompute_residual = True
    
    
    # Experimental
    
    wexp_shift = -0.5
    
    # what to plot: 'radiance' or 'transmittance'
    plotquantity = 'radiance'
    unit = 'mW/cm2/sr/nm'
    #unit = 'default'
    
#    fexp = r"data\12_StepAndGlue_30us_Cathode_0us_stacked.txt"
    fexp = getTestFile(r"measured_co2_bandhead_10kHz_30us.txt")
    wexp, Iexp = np.loadtxt(fexp, skiprows=1).T

    
    from .tools import Normalizer
    normalizer = None #Normalizer(4173, 4180, how='mean')
    
    # -----------------------------------------------------------------------------
    # NON USER PARAM PART
    # -----------------------------------------------------------------------------
    
    dbInteractx = Slablist[slbInteractx]['db']
    dbInteracty = Slablist[slbInteracty]['db']
            
    import warnings
    warnings.filterwarnings("ignore", "Using default event loop until function specific"+\
                            "to this GUI is implemented")
    
    
    fitroom = FitRoom(Slablist)
    
    solver = SlabsConfigSolver(config=config, source='database',
                               wexp=wexp, Iexpcalib=Iexp, wexp_shift=wexp_shift,
                               plotquantity=plotquantity, unit=unit,
                               slit=slit)
    
    gridTool = Grid3x3(slbInteractx=slbInteractx, slbInteracty=slbInteracty,
                       xparam=xparam, yparam=yparam, 
                       plotquantity=plotquantity, unit=unit,
                       normalizer=normalizer,
                       wexp=wexp, Iexpcalib=Iexp, wexp_shift=wexp_shift)
    
    slabsTool = MultiSlabPlot(plotquantity=plotquantity, unit=unit,
                              normalizer=normalizer,
                              wexp=wexp, Iexpcalib=Iexp, wexp_shift=wexp_shift,
                              nfig=3)
    
    selectTool = CaseSelector(dbInteractx, dbInteracty, xparam, yparam, 
                              slbInteractx=slbInteractx, slbInteracty=slbInteracty,  
                              nfig=1)
    
    slitTool = SlitTool()
    
#    overpTool = Overpopulator(overpSlab)
    
    fitroom.add_tool(solver)
    fitroom.add_tool(gridTool)
    fitroom.add_tool(slabsTool)
    fitroom.add_tool(selectTool)
#    fitroom.add_tool(overpTool)
    fitroom.add_tool(slitTool)
    
    
    
    # Map x, y
    # -----------
    xvar = Slablist[slbInteractx][xparam]
    yvar = Slablist[slbInteracty][yparam]
    
    xspace = linspace(xvar*(1-xstep), xvar*(1+xstep), 3)
    yspace = linspace(yvar*(1-ystep), yvar*(1+ystep), 3)
    
    gridTool.plot_3times3(xspace, yspace)
    
    
    # TODO: precompute not working on example. Fix it
#    # 2D mapping
#    if precompute_residual:
#        selectTool.precompute_residual(Slablist)
    
