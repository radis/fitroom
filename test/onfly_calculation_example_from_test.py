# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:42:13 2021

@author: erwan, CorentinGrimaldi, BlehMaks

Description
-----

Functions to build a 2D, multislabs fitting room with a pure Python (Matplotlib)
interface that displays:

    (1) a window to select conditions along two axis (to calculate, or retrieve
    from a database)

    (2) a window to plot 9 spectra corresponding to left, right and center conditions

    (3) a window to decompose the center slab along the different slabs

This example calculates spectra on-fly.

Todo
-----

interface
 - used keyboards keys to move rectangle selector
"""

import warnings
from radis.los.slabs import SerialSlabs#, MergeSlabs
import numpy as np
from numpy import linspace
from fitroom import CaseSelector
from fitroom import Grid3x3
from fitroom import MultiSlabPlot
from fitroom import SlabsConfigSolver
from fitroom import FitRoom, DynVar
from fitroom import SlitTool
#from fitroom.tools import Normalizer
from radis import SpectrumFactory
from radis import Spectrum
from os.path import join, dirname

TEST_FOLDER_PATH = dirname(__file__)

def getTestFile(file):
    ''' Return the full path of a test file. Used by test functions not to
    worry about the project architecture'''

    return join(TEST_FOLDER_PATH, file)

# %% Load Database
# -----------------------------------------------------------------------------
wav_min = 4165
wav_max = 4200

sf2 = SpectrumFactory(wavelength_min=wav_min,
                      wavelength_max=wav_max,
                      cutoff=1e-25,
                      isotope=[1, 2],
                      medium='air')
sf2.load_databank('HITRAN-CO2-TEST')
sf2.init_database(getTestFile('HITRAN_CO2_test_spec_database'))
# %% `Plot fit

slbPlasmaCO2 = {'factory': sf2,
                'Tgas': 1500,
                #         'Tvib':1100,
                #         'Trot':1200,
                'path_length': 0.02,
                'mole_fraction': 0.6,
                }

slbPostCO2 = {'factory': sf2,
              'Tgas': 350,
              'path_length': 0.7,
              #'mole_fraction':DynVar('sPlasmaCO2', 'mole_fraction'),
              # TODO: line above doesnt work with precompute residual. Fix it!
              'mole_fraction': 1,
              }

slbRoomCO2 = {'factory': sf2,
              'Tgas': 300,
              'path_length': 373,  # cm
              'mole_fraction': 400e-6,
              }

Slablist = {'sPlasmaCO2': slbPlasmaCO2,
            'sPostCO2': slbPostCO2,
            'sRoomCO2': slbRoomCO2}


def config(**slabs):
    ''' args must correspond to slablist. Indexes and order is important '''
    return SerialSlabs(slabs['sPlasmaCO2'], slabs['sPostCO2'], slabs['sRoomCO2'])

# Sensibility analysis

slbInteractx = 'sPlasmaCO2'
xparam = 'Tgas'
slbInteracty = 'sPlasmaCO2'
yparam = 'mole_fraction'
xstep = 0.2
ystep = 0.2

# Slit
slit = (1.63, 2.33)

# what to plot: 'radiance' or 'transmittance'
plotquantity = 'radiance'
unit = 'mW/cm2/sr/nm'
wunit = 'nm'
#unit = 'default'

s_exp = Spectrum.from_txt(getTestFile(r"measured_co2_bandhead_10kHz_30us.txt"),
                          quantity=plotquantity, waveunit=wunit, unit=unit,
                          conditions={'medium': 'air'}).offset(-0.5, 'nm')
normalizer = None  # Normalizer(4173, 4180, how='mean')

# -----------------------------------------------------------------------------
# NON USER PARAM PART
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore", "Using default event loop until function specific" +
                        "to this GUI is implemented")

fitroom = FitRoom(Slablist, slbInteractx, slbInteracty, xparam, yparam)

solver = SlabsConfigSolver(config=config, source='calculate',
                           s_exp=s_exp,
                           plotquantity=plotquantity, unit=unit,
                           slit=slit)
gridTool = Grid3x3(slbInteractx=slbInteractx, slbInteracty=slbInteracty,
                   xparam=xparam, yparam=yparam,
                   plotquantity=plotquantity, unit=unit,
                   normalizer=normalizer,
                   s_exp=s_exp)
slabsTool = MultiSlabPlot(plotquantity=plotquantity, unit=unit,
                          normalizer=normalizer,
                          s_exp=s_exp,
                          nfig=3)
selectTool = CaseSelector(xparam=xparam, yparam=yparam,
                          slbInteractx=slbInteractx, slbInteracty=slbInteracty,
                          nfig=1)
slitTool = SlitTool('nm')  # getTestFile('slitfunction.txt'))

fitroom.add_tool(solver)
fitroom.add_tool(gridTool)
fitroom.add_tool(slabsTool)
fitroom.add_tool(selectTool)


# Map x, y
# -----------
xvar = Slablist[slbInteractx][xparam]
yvar = Slablist[slbInteracty][yparam]

xspace = linspace(xvar*(1-xstep), xvar*(1+xstep), 3)
yspace = linspace(yvar*(1+ystep), yvar*(1-ystep), 3)

gridTool.plot_3times3(xspace, yspace)

fitroom.add_tool(slitTool)

select_xspace = (xspace.min()*0.9, xspace.max()*1.1)
select_yspace = (yspace.min()*0.9, yspace.max()*1.1)

selectTool.ax.set_xlim(*select_xspace)
selectTool.ax.set_ylim(*select_yspace)

# Precompute residual for all points
# --------------

selectTool.precompute_residual(Slablist, # plotquantity='radiance',
                               xspace=np.linspace(300, 2000, 3),
                               yspace=np.linspace(0, 1, 3))