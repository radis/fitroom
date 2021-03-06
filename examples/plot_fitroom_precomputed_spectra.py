# -*- coding: utf-8 -*-
"""

====================================
Use Fitroom with precomputed spectra
====================================

Compare experimental spectrum to a combination of precomputed spectra
powered by the RADIS :py:class:`~radis.tools.database.SpecDatabase` class



Description
-----------

Functions to build a 2D, multislabs fitting room with a pure Python (Matplotlib)
interface that displays:

- :py:class:`~fitroom.selection_tool.CaseSelector` : a window to select conditions along two axis (to calculate, or retrieve
    from a database)

- :py:class:`~fitroom.grid3x3_tool.Grid3x3` : a window to plot 9 spectra corresponding to left, right and center conditions

- :py:class:`~fitroom.multislab_tool.MultiSlabPlot` : a window to decompose the center slab along the different slabs

"""

from radis import Spectrum
from radis.tools.database import SpecDatabase
from radis.los.slabs import SerialSlabs
from numpy import linspace
from publib import set_style
from radis.test.utils import setup_test_line_databases
from fitroom import CaseSelector
from fitroom import Grid3x3
from fitroom import MultiSlabPlot
from fitroom import SlabsConfigSolver
from fitroom import FitRoom
from fitroom import SlitTool
from radis import SpectrumFactory

from os.path import join, dirname

import matplotlib.pyplot as plt
plt.ion()   # interactive mode; do not block figures

TEST_FOLDER_PATH = dirname(".")

def getTestFile(file):
    ''' Return the full path of a test file. Used by test functions not to
    worry about the project architecture'''

    return join(TEST_FOLDER_PATH, file)


def _generate_test_database():

    # %% Generate test spectra database (just for the example)
    # ------------------------

    # Add HITRAN-TEST-CO2 to neq.rc (create file if doesnt exist)
    setup_test_line_databases()

    sf2 = SpectrumFactory(
        wavelength_min=4165,
        wavelength_max=4200,
        mole_fraction=1,
        path_length=0.025,
        cutoff=1e-25,
        isotope=[1, 2],
        db_use_cached=True,
        medium='air')
    sf2.load_databank('HITRAN-CO2-TEST')
    sf2.init_database(getTestFile(
        'HITRAN_CO2_test_spec_database'), autoretrieve=True)  # 'force')

    for Tgas in [300, 350, 400, 1200, 1300, 1500, 1700, 2000]:
        sf2.eq_spectrum(Tgas)
        # Note that autoretrieve is set to True by default, so already generated
        # spectra wont be calculated again


_generate_test_database()

set_style('origin')

# %% Load Database
# -----------------------------------------------------------------------------

db0 = SpecDatabase(getTestFile('HITRAN_CO2_test_spec_database'), lazy_loading=False, nJobs=1)   # CO2

# %% `Plot fit

slbPlasmaCO2 = {
    'db': db0,
    'Tgas': 1500,
    #         'Tvib':1100,
    #         'Trot':1200,
    'path_length': 0.02,
    'mole_fraction': 0.6,
}

slbPostCO2 = {
    'db': db0,
    'Tgas': 350,
    'path_length': 0.7,
    # 'mole_fraction':DynVar('sPlasmaCO2', 'mole_fraction'),
    # TODO: line above doesnt work with precompute residual. Fix it!
    'mole_fraction': 1,
}

slbRoomCO2 = {
    'db': db0,
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
#slbInteract = slbPostCO
#xparam = 'Tgas'
#yparam = 'path_length'
xstep = 0.2
ystep = 0.2

slit = (1.63, 2.33)

# Experimental

#    wexp_shift = -0.5

# what to plot: 'radiance' or 'transmittance'
plotquantity = 'radiance'
unit = 'mW/cm2/sr/nm'
#unit = 'default'

s_exp = Spectrum.from_txt(getTestFile(r"measured_co2_bandhead_10kHz_30us.txt"), wunit='nm', 
                          quantity=plotquantity, unit='mW/cm2/sr/nm',
                          conditions={'medium': 'air'}).offset(-0.5, 'nm')

normalizer = None  # Normalizer(4173, 4180, how='mean')

# -----------------------------------------------------------------------------
# NON USER PARAM PART
# -----------------------------------------------------------------------------

dbInteractx = Slablist[slbInteractx]['db']
dbInteracty = Slablist[slbInteracty]['db']

import warnings
warnings.filterwarnings("ignore", "Using default event loop until function specific" +
                        "to this GUI is implemented")

fitroom = FitRoom(Slablist, slbInteractx, slbInteracty, xparam, yparam)

solver = SlabsConfigSolver(config=config, source='database',
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

selectTool = CaseSelector(dbInteractx, dbInteracty, xparam, yparam,
                          slbInteractx=slbInteractx, slbInteracty=slbInteracty,
                          nfig=1)

slitTool = SlitTool('nm')  # getTestFile('slitfunction.txt'))

#    overpTool = Overpopulator(overpSlab)

fitroom.add_tool(solver)
fitroom.add_tool(gridTool)
fitroom.add_tool(slabsTool)
fitroom.add_tool(selectTool)
#    fitroom.add_tool(overpTool)


# Map x, y
# -----------
xvar = Slablist[slbInteractx][xparam]
yvar = Slablist[slbInteracty][yparam]

xspace = linspace(xvar*(1-xstep), xvar*(1+xstep), 3)
yspace = linspace(yvar*(1-ystep), yvar*(1+ystep), 3)

gridTool.plot_3times3(xspace, yspace)
fitroom.add_tool(slitTool)
# Precompute residual for all points
# --------------

selectTool.precompute_residual(Slablist, yspace=[1e-3, 0.1, 0.5, 1, 2])



plt.show()