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

#DEBUG. TEST CASE increase mole_fraction then decrease

from neq.spec.database import SpecDatabase
from neq.spec.calc import MergeSlabs, SerialSlabs
import matplotlib.pyplot as plt 
import numpy as np
from numpy import linspace, array
from matplotlib.widgets import MultiCursor
from scipy.interpolate import griddata, splev, splrep
from publib import set_style
from neq.math.smooth import als_baseline

from selection_tool import CaseSelector
from grid3x3_tool import Grid3x3

set_style('origin')

# %% Load Database
# -----------------------------------------------------------------------------

dbp = SpecDatabase('co2SpecDatabase')   # plasma
dbpco = SpecDatabase('CO_SpecDatabase_noneq1iso')   # plasma CO
db0 = SpecDatabase('co2SpecDatabase_eq')   # room
dbco = SpecDatabase(r'coSpecDatabase_eq')  # post discharge CO 
dbh = SpecDatabase(r'H2O_SpecDatabase_eq')   # room
#
## Fix nans
#for sp in dbpco.get():
#    w, I = sp['radiance_noslit']
#    I[np.isnan(I)] = 0
#    sp['radiance_noslit'] = w, I
      

# %% `Plot fit




slbPlasmaCO2 = {
         'db':dbp,
         'Tvib':1200,
         'Trot':1550,
         'path_length':0.025,
         'mole_fraction':1,
         }

slbPlasmaCO = {
         'db':dbpco,
         'Tvib':1300,
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
#                       MergeSlabs(slabs['sPlasmaCO2'], slabs['sPlasmaCO'], accept_different_lengths=True), 
                       MergeSlabs(sBaseline, accept_different_lengths=True), 
                       MergeSlabs(slabs['sPlasmaCO'], accept_different_lengths=True), 
                       MergeSlabs(slabs['sPostCO2'], slabs['sPostCO'], accept_different_lengths=True),
                       MergeSlabs(slabs['sRoomCO2'], slabs['sRoomH2O'], accept_different_lengths=True),
                       )

# Sensibility analysis

slbInteractx = 'sPlasmaCO'
xparam = 'Tvib'
slbInteracty = 'sPlasmaCO'
yparam = 'Trot'
#slbInteract = slbPostCO
#xparam = 'Tgas'
#yparam = 'path_length'
xstep = 0.2
ystep = 0.2
slit = 1.5  # r"D:\These Core\1_Projets\201702_CO2_insitu\20170505_TeteDeBandeTrot\Calibration\slit_spectrum_cleaned.txt" #  1.7



verbose=False
normalize = False
precompute_residual = False


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
#    
    # TODO: calibrer


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

# -----------------------------------------------------------------------------

dbInteractx = Slablist[slbInteractx]['db']
dbInteracty = Slablist[slbInteracty]['db']


from tools import Normalizer
norm_on = Normalizer(4173, 4180, how='mean')

#Lroom = 100
#nC02room = 400e-6
# ... look around with step
#Tvibstep = 0.05
#Trotstep = 0.1

# Print score for 2D mapping
def get_residual(s, norm='not_implemented'):
    ''' Different between experimental and simulated spectra 
    
    norm not implemented yet 
    # TODO
    
    Implementation
    -------

    interpolate experimental is harder (because of noise, and overlapping)
    we interpolate each new spectrum on the experiment
    
    '''
    
    b = np.argsort(wexp)
    wsort, Isort = wexp[b], Iexpcalib[b]
    
    
    w, I = s.get(plotquantity, xunit='nm', yunit=unit)
    w, I = w[::-1], I[::-1]
    
    tck = splrep(w, I)
    
    Iint = splev(wsort, tck)
    
#    error = np.sqrt(np.trapz(np.abs((Iint-Isort)/(Iint+Isort)), x=wsort).sum())
    error = np.sqrt(np.trapz(np.abs(Iint-Isort), x=wsort).sum())
#    error = np.sqrt(((Ixp-I)**2).sum())


    return error


# Generate Figure Layout
    
# Plot database
def plot_params(dbInteractx, xparam, dbInteracty, yparam, nfig=None): 
    x = dbInteractx.df[xparam]
    y = dbInteracty.df[yparam]

    # Plot
    fig, ax = plt.subplots(num=nfig)
    if dbInteractx == dbInteracty:
        ax.plot(x, y, 'ok')
        
    else:
        xx, yy = np.meshgrid(list(set(x)), list(set(y)))
        ax.plot(xx, yy, 'ok')
    # TODO: add units from spectrum here. (but maybe units arent the same for all database????)
    # load it up first and check? 
    ax.set_xlabel('{0} {1}'.format(slbInteracty, xparam))   # flipped x y
    ax.set_ylabel('{0} {1}'.format(slbInteractx, yparam))
    
    return fig, ax

def calc_slabs(**slabsconfig):
    ''' 
    Input
    ------
    
    slabsconfig: 
        list of dictionaries. Each dictionary as a database key `db` and 
        as many conditions to filter the database

    '''

    slabs = {}  # slabs
    fconds = {}   # closest conditions found in database
    
    for slabname, slabcfg in slabsconfig.items():
        cfg = slabcfg.copy()
        dbi = cfg.pop('db')
#        cfg['verbose'] = verbose

        si = dbi.get_closest(scale_if_possible=True, verbose=verbose, **cfg)
#        try:
#            del slabcfg['verbose']
#        except KeyError:
#            pass

        fcondsi = {}
        for k in cfg:
            fcondsi[k] = si.conditions[k]
        slabs[slabname] = si.copy()
        fconds[slabname] = fcondsi

    s = config(**slabs)
    s.apply_slit(slit, norm_by='area', shape='triangular')
    
    return s, slabs, fconds
    



plt.close(1)
fig1, ax1 = plot_params(dbInteracty, yparam, dbInteractx, xparam, nfig=1)   # yeah, i flipped it -_-

gridTool = Grid3x3(calc_slabs=calc_slabs, 
                   slbInteractx=slbInteractx, slbInteracty=slbInteracty, 
                   xparam=xparam, yparam=yparam,
                   plotquantity=plotquantity, unit=unit,
                   normalize=False, normalizer=norm_on,
                   wexp=wexp, Iexpcalib=Iexpcalib, wexp_shift=wexp_shift,
                   get_residual=get_residual,
                   MultiSlabPlot=None,
                   CaseSelector=None,
                   Slablist=Slablist)

fig2 = gridTool.fig
ax2 = gridTool.ax
    
# Generate Figure 3 layout
plt.figure(3, figsize=(12,8)).clear()
fig3, ax3 = plt.subplots(3, 1, num=3, sharex=True)
line3up = {}
line3cent = {}
line3down = {}
plt.tight_layout()


##    s0 = db0.get(path_length=Lroom, Tgas=Troom)[0] 
#    s0 = db0.get_closest(path_length=Lroom, Tgas=Troom, mole_fraction=1, verbose=verbose)
#    s1 = db0.get_closest(path_length=Lroom1, Tgas=Troom1, verbose=verbose)
#    s2 = db0.get_closest(path_length=Lroom2, Tgas=Troom2, mole_fraction=408e-6, verbose=verbose)
#    sCO = dbco.get_closest(path_length=6, Tgas=TgasCO, mole_fraction=0.03, verbose=verbose)
#    sH2O = dbh.get_closest(path_length=373, Tgas=300, mole_fraction=0.02, verbose=verbose)
    

# ARBITRARY... based on Tungsten lamp being 20% strong than the other
# Todo. Check. Don't take for granted.
#if True:
##    Iexpcalib /= 1.2
#    Iexpcalib /= 1
#    print('Warning. Tungsten 20% stronger hypothesis')

def plot_all_slabs(s, slabs):
    
    # Central axe: model vs experiment
    w, I = s.get(plotquantity)
    ydata = norm_on(w, I) if normalize else I
    try:
        line3cent[1].set_data(w, ydata)
#        line3cent[1].set_data(*s.get('radiance'))
    except KeyError:
        line3cent[1] = ax3[1].plot(w, ydata, color='r', lw=1, 
                 label='Model')[0]
        ax3[1].set_ylabel(s.units[plotquantity])
        
    ydata = norm_on(wexp, Iexpcalib) if normalize else Iexpcalib
    try:        
        line3cent[0].set_data(wexp+wexp_shift, ydata)
    except KeyError:
        line3cent[0] = ax3[1].plot(wexp+wexp_shift, ydata,'-k', lw=1, zorder=-1, 
                 label='Experiment')[0]
        ax3[1].legend()
        
    def colorserie():
        i = 0
        colorlist = ['r', 'b', 'g', 'y', 'k', 'm']
        while True:
            yield colorlist[i%len(colorlist)]
            i += 1
            
    # Upper axe: emission   &    lower axe: transmittance
    try:
        colors = colorserie()
        for i, (name, s) in enumerate(slabs.items()):
            s.apply_slit(slit)
            color = next(colors)
            line3up[i].set_data(*s.get('radiance'))
            line3down[i].set_data(*s.get('transmittance'))
    except KeyError:  # first time: init lines
        colors = colorserie()
        for i, (name, si) in enumerate(slabs.items()):
            si.apply_slit(slit)
            color = next(colors)
            line3up[i] = ax3[0].plot(*si.get('radiance'), color=color, lw=2, 
                   label=name)[0]
            line3down[i] = ax3[2].plot(*si.get('transmittance'), color=color, lw=2, 
                     label=name)[0]
        if not normalize: 
            ax3[2].set_ylim((-0.008, 0.179)) # Todo: remove that 
        ax3[2].set_ylim((0,1))
        ax3[0].legend()
        ax3[2].legend()
        ax3[0].xaxis.label.set_visible(False)
        ax3[1].xaxis.label.set_visible(False)
        ax3[2].set_xlabel(s.wavespace())
        ax3[0].set_ylabel(si.units['radiance'])
        ax3[2].set_ylabel(si.units['transmittance'])
        fig3.tight_layout()

# Map x, y
# -----------
xvar = Slablist[slbInteractx][xparam]
yvar = Slablist[slbInteracty][yparam]

xspace = linspace(xvar*(1-xstep), xvar*(1+xstep), 3)
yspace = linspace(yvar*(1-ystep), yvar*(1+ystep), 3)


gridTool.plot_3times3(xspace, yspace)

# Database inspect
# -------------------------------------



multi2 = MultiCursor(fig2.canvas, (*ax2[0], *ax2[1], *ax2[2]), color='r', lw=1,
                    alpha=0.2, horizOn=False, vertOn=True)

multi3 = MultiCursor(fig3.canvas, (ax3[0], ax3[1], ax3[2]), color='r', lw=1,
                    alpha=0.2, horizOn=False, vertOn=True)
#plt.figure(2)
#plt.xlim((4455.5123188889647, 4559.3683455836699))
#plt.ylim(ymin=0)

# %% 2D mapping

if precompute_residual:
    # Plot residual for all points in database
    
#    import warnings
#    with warnings.catch_warnings():
#        warnings.filterwarnings('error')

    if dbInteractx == dbInteracty and False:
        ''' Doesnt work... fix later? 
        I think it doesnt like the sorting
        '''
        
        
        # only calculate database points 
        xspace, yspace = zip(*array(dbInteractx.view()[[xparam, yparam]]))
        # kill duplicates
        xspace, yspace = zip(*set(list(zip(xspace, yspace))))

        xx, yy = np.meshgrid(xspace, yspace)        
        
        res = []  #np.empty_like(xx)
            
        for xvari, yvarj in zip(xspace, yspace):
            config0 = {k:c.copy() for k, c in Slablist.items()}
            config0[slbInteractx][xparam] = xvari
            config0[slbInteracty][yparam] = yvarj
                
        #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"

            s, slabs, fconfig = calc_slabs(**config0)
                    
            resij = get_residual(s)
            
            print(xparam, xvari, yparam, yvarj, resij)
                
            res.append(resij)  # yes flipped it
            
        res = array(res)
        # Create a 2D grid by interpolating database data
        res = griddata((yspace, xspace), res, (yy, xx))   # yes flipped it

    else:
        # do a mapping of all possible cases
        xspace = array(sorted(set(dbInteractx.view()[xparam])))
        yspace = array(sorted(set(dbInteracty.view()[yparam])))
        
        xx, yy = np.meshgrid(xspace, yspace)        
        
        res = np.empty_like(xx)
            
        for i, xvari in enumerate(xspace):
            for j, yvarj in enumerate(yspace):
                config0 = {k:c.copy() for k, c in Slablist.items()}
                config0[slbInteractx][xparam] = xvari
                config0[slbInteracty][yparam] = yvarj
                    
            #        fexp = r"12_StepAndGlue_30us_Cathode_0us_stacked.txt"
    
                s, slabs, fconfig = calc_slabs(**config0)
                        
                resij = get_residual(s)

                print(xparam, xvari, yparam, yvarj, resij)
                    
                res[j][i] = resij  # yes flipped it

    cf = ax1.contourf(yy, xx, res, 40, cmap=plt.get_cmap('viridis_r'))  # flipped it
    cbar = fig1.colorbar(cf)
    cbar.ax.set_ylabel('residual')
    plt.tight_layout()

# %%

selectTool = CaseSelector(ax1=ax1, fig2=fig2, fig3=fig3, gridTool=gridTool)
gridTool.CaseSelector = selectTool


# %%
#
#slit = params['slit']
#
#
#plt.figure(10).clear()
#
#plt.plot(wexp+1, Iexpcalib/Iexpcalib.max()*0.01, 'k')
#s.plot('radiance', nfig=10, c='b')
#
#sco = dbco.get('mole_fraction == 0.01 & Tgas == 350')
#assert(len(sco)==1)
#sco = sco[0]

#
#st = ETR(s, sco)
#st.apply_slit(slit)
#st.plot('radiance', nfig=10, c='r')
#

#sco.apply_slit(slit)
#sco.plot('radiance', nfig=10, c='b')







# %% -----
