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

from neq.spec.database import SpecDatabase
from neq.spec.calc import MergeSlabs, SerialSlabs
import numpy as np
from numpy import linspace
from scipy.interpolate import splev, splrep
from publib import set_style
from neq.math.smooth import als_baseline

try:
    from neq.math.fitroom import CaseSelector
    from neq.math.fitroom import Grid3x3
    from neq.math.fitroom import MultiSlabPlot
    from neq.math.fitroom import SlabsConfigSolver
    from neq.math.fitroom import Overpopulator
    from neq.math.fitroom import FitRoom
except:
    from .selection_tool import CaseSelector
    from .grid3x3_tool import Grid3x3
    from .multislab_tool import MultiSlabPlot
    from .solver import SlabsConfigSolver
    from .noneq_tool import Overpopulator
    from .room import FitRoom
    
from neq.spec import SpectrumFactory
from neq.spec.bands import BandList
    
if __name__ == '__main__':
    
    set_style('origin')
    
    # %% Load Database
    # -----------------------------------------------------------------------------
    
#    dbp = SpecDatabase('co2SpecDatabase')   # plasma
#    dbpco = SpecDatabase('CO_SpecDatabase_noneq1iso')   # plasma CO
#    db0 = SpecDatabase('co2SpecDatabase_eq')   # room
#    dbco = SpecDatabase(r'coSpecDatabase_eq')  # post discharge CO
#    dbh = SpecDatabase(r'H2O_SpecDatabase_eq')   # room
##    
#    # Fix nans
#    for sp in dbpco.get():
#        w, I = sp['radiance_noslit']
#        I[np.isnan(I)] = 0
#        sp['radiance_noslit'] = w, I
        
    
#    # Test 
#    sf2 = SpectrumFactory(
#                         wavelength_min=4170,
#                         wavelength_max=4200,
#                         mole_fraction=1,
#                         path_length=0.025, 
#                         cutoff=1e-25,
#                         isotope_identifier=[1,2],
#                         use_cached=True,
#                         medium='air')
#    sf2.load_databank('CDSD')
#    s_bands2 = sf2.non_eq_bands(Tvib=Tvib, Trot=1500) #, return_lines=False)
#    bandlist = BandList(s_bands2)
    
#    bandlist.set_cutoff(1e-4)

    
    # %% `Plot fit
    
    
    bandlistCO2 = bandlist
    
    slbPlasmaCO2 = {
             'db':dbp,
             'bandlist':bandlistCO2,
             'Tvib':2500,
             'Trot':1500,
             'path_length':0.025,
             'mole_fraction':1,
             'source':'from_bands',
             }
    
    slbPlasmaCO = {
             'db':dbpco,
             'Tvib':2500,
             'Trot':1450,
             'path_length':0.025,
             'mole_fraction':0.4,
             }
    
    slbPostCO2 = {
             'db':db0,
             'Tgas':375,
    #         'db':dbp,
    #         'Tvib':1100,
    #         'Trot':1200,
             'path_length':5.5,
             'mole_fraction':1,
             }
    
    slbPostCO = {
             'db':dbco,
             'Tgas':350,
             'path_length':5.5,
             'mole_fraction':0.02,
             }
    
    slbRoomCO2 = {
             'db':db0,
             'Tgas':300,
             'path_length':373, #+20,
             'mole_fraction':400e-6,
             }
    
    slbRoomH2O = {
            'db':dbh,
             'path_length':373,
             'Tgas':300,
             'mole_fraction':0.02,
             }
    
    Slablist = {'sPlasmaCO2':slbPlasmaCO2,
                'sPlasmaCO':slbPlasmaCO,
                'sPostCO2': slbPostCO2,
                'sPostCO': slbPostCO,
                'sRoomCO2': slbRoomCO2,
                'sRoomH2O': slbRoomH2O}
    
    def config(**slabs):
        ''' args must correspond to slablist. Indexes and order is important '''
    
    #    return SerialSlabs(sPlasmaCO2, sPlasmaCO, sPostCO2, sPostCO, sRoomCO2, sRoomH2O)
    
        globals().update({'spectra':slabs})
    
        return SerialSlabs(
                           MergeSlabs(slabs['sPlasmaCO2'], resample_wavespace='full'),
#                           MergeSlabs(sBaseline, accept_different_lengths=True),
                           MergeSlabs(slabs['sPostCO2'], slabs['sPostCO'], resample_wavespace='full'),
                           MergeSlabs(slabs['sRoomCO2'], slabs['sRoomH2O'], resample_wavespace='full'),
                           resample_wavespace='full')
    
    # Sensibility analysis
    
    slbInteractx = 'sPlasmaCO2'
    xparam = 'Tvib'
    slbInteracty = 'sPlasmaCO2'
    yparam = 'mole_fraction'
    #slbInteract = slbPostCO
    #xparam = 'Tgas'
    #yparam = 'path_length'
    xstep = 0.2
    ystep = 0.2
    slit = 1.5  # r"D:\These Core\1_Projets\201702_CO2_insitu\20170505_TeteDeBandeTrot\Calibration\slit_spectrum_cleaned.txt" #  1.7
    
    # Add overpopulation on given slab
    overpSlab = slbPlasmaCO2
    
    
    verbose=False
    normalize = False
    precompute_residual = True
    
    
    # Experimental
    
    wexp_shift = 1
    
    # what to plot: 'radiance' or 'transmittance'
    plotquantity = 'radiance'
    unit = 'mW/cm2/sr/nm'
    #unit = 'default'
    
    if plotquantity == 'radiance':
        fexp = r"data\15_20kHz_StepAndGlue_10us_jetCathode_Cathode_t0us_avg_stacked.txt"
    #    fexp = r"data\14_20kHz_StepAndGlue_30us_jetCathode_Cathode_t20us_avg_stacked.txt"
        wexp, Iexp = np.loadtxt(fexp).T
    
        # Absolute calibration
        wglo, Iglo = np.loadtxt(r"D:\These Core\1_Projets\201702_CO2_insitu\20170505_TeteDeBandeTrot\Calibration\calibOL550\calibration_globarThorlabs.txt",
                                skiprows=2, delimiter=',').T  # calib globar
        tck = splrep(wglo, Iglo)
        Icalib = splev(wexp, tck)
        from neq import planck
        Ioldcalib = planck(wexp, 1500, eps=1, unit=unit)   # formerly calibrated with planck eps=1, T=1500K
        Iexpcalib = Iexp / Ioldcalib * Icalib   # ▓calibrated in absolute
    
        # ARBITRARY... based on Tungsten lamp being 20% strong than the other
        # Todo. Check. Don't take for granted.
        #if True:
        ##    Iexpcalib /= 1.2
        #    Iexpcalib /= 1
        #    print('Warning. Tungsten 20% stronger hypothesis')
    
    
    elif plotquantity == 'transmittance':
    
        # Note: we correctly measure transmittance only if blackbody ~ constant on
        # experimental slit length (because T = (I/I0)_slit !=  I_slit/I0_slit in
        # general )
    
        fexp = r"data\2_CathodeJet_20kHz_Vref1_3mmBelowCat_stacked.txt"
        #fexp = r"14_20kHz_StepAndGlue_30us_jetCathode_Cathode_t20us_avg_stacked.txt"
        wexp, Iexp = np.loadtxt(fexp).T
        fexpref = r"data\8_Reference_atmAir_stacked.txt"
        wexpref, Iexpref = np.loadtxt(fexpref).T
    
    
    #    fexp2 = r"data\7_CathodeJet_20kHz_Vref1_5mmAboveCat_stacked.txt"
    #    wexp2, Iexp2 = np.loadtxt(fexp2).T
    #
    #    Iem = Iexp - Iexp2
    #
    #    assert((wexp==wexpref).all())
    #    Iexpcalib = Iexp/Iexpref
    
    # %% Get baseline
    
    from neq.spec import theoretical_spectrum
    fexp = r"data\15_20kHz_StepAndGlue_10us_jetCathode_Cathode_t0us_avg_stacked.txt"
    #    fexp = r"data\14_20kHz_StepAndGlue_30us_jetCathode_Cathode_t20us_avg_stacked.txt"
    wexp, Iexp = np.loadtxt(fexp).T
    b = np.argsort(wexp)
    wsort, Isort = wexp[b], Iexpcalib[b]
    Ismooth = als_baseline(Isort, 0.8, 1e8)
    tck = splrep(wsort+wexp_shift, Ismooth)
    wbas = dbp.get()[0].get('radiance_noslit')[0]
    Ibas = splev(wbas, tck)
    sBaseline = theoretical_spectrum(wbas, Ibas, wunit='nm', Iunit='mW/cm2/sr/nm')
    
    
    from tools import Normalizer
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
                               wexp=wexp, Iexpcalib=Iexpcalib, wexp_shift=wexp_shift,
                               plotquantity=plotquantity, unit=unit,
                               slit=slit)
    
    gridTool = Grid3x3(slbInteractx=slbInteractx, slbInteracty=slbInteracty,
                       xparam=xparam, yparam=yparam, 
                       plotquantity=plotquantity, unit=unit,
                       normalizer=normalizer,
                       wexp=wexp, Iexpcalib=Iexpcalib, wexp_shift=wexp_shift)
    
    slabsTool = MultiSlabPlot(plotquantity=plotquantity, unit=unit,
                              normalizer=normalizer,
                              wexp=wexp, Iexpcalib=Iexpcalib, wexp_shift=wexp_shift,
                              nfig=3, slit=slit)
    
    selectTool = CaseSelector(dbInteractx, dbInteracty, xparam, yparam, 
                              slbInteractx=slbInteractx, slbInteracty=slbInteracty,  
                              nfig=1)
    
    overpTool = Overpopulator(overpSlab)
    
    fitroom.add_tool(solver)
    fitroom.add_tool(gridTool)
    fitroom.add_tool(slabsTool)
    fitroom.add_tool(selectTool)
    fitroom.add_tool(overpTool)
    
    
    
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
    
