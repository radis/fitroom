# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:32:23 2017

@author: erwan

Machinery related to solving a specific Slabs configuration: parse the database,
get the correct slab input, then calls the appropriate functions in neq.spec engine

"""

import numpy as np
from scipy.interpolate import splev, splrep

class SlabsConfigSolver():
    '''
    Machinery related to solving a specific Slabs configuration: parse the database,
    get the correct slab input, then calls the appropriate functions in neq.spec engine
    '''

    def __init__(self, config, mode=None,
                 wexp=None, Iexpcalib=None, wexp_shift=0,
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 slit=None, slit_options={'norm_by':'area', 'shape':'triangular'},
                 verbose=True):

#        self.dbInteractx = dbInteractx
#        self.dbInteracty = dbInteracty

        self.config = config

        self.wexp = wexp
        self.Iexpcalib = Iexpcalib
        self.wexp_shift = wexp_shift
        self.plotquantity = plotquantity
        self.unit = unit
        self.slit = slit
        self.slit_options = slit_options

        self.verbose = verbose
        
        
        self.mode = mode

    def get_residual(self, s, norm='not_implemented'):
        ''' Different between experimental and simulated spectra

        norm not implemented yet
        # TODO

        Implementation
        -------

        interpolate experimental is harder (because of noise, and overlapping)
        we interpolate each new spectrum on the experiment

        '''

        wexp = self.wexp
        Iexpcalib = self.Iexpcalib
        plotquantity = self.plotquantity
        unit = self.unit


        b = np.argsort(wexp)
        wsort, Isort = wexp[b], Iexpcalib[b]

        w, I = s.get(plotquantity, xunit='nm', yunit=unit)
        
        # crop to overlapping range
        b = (wsort>w.min()) & (wsort<w.max())
        wsort, Isort = wsort[b], Isort[b]
        b = (w>wsort.min()) & (w<wsort.max())
        w, I= w[b], I[b]
        
        if w[0]>w[-1]:
            w, I = w[::-1], I[::-1]

        tck = splrep(w, I)
        Iint = splev(wsort, tck)

    #    error = np.sqrt(np.trapz(np.abs((Iint-Isort)/(Iint+Isort)), x=wsort).sum())
        error = np.sqrt(np.trapz(np.abs(Iint-Isort), x=wsort).sum())
        
        return error


    def calc_slabs(self, **slabsconfig):
        '''
        Input
        ------

        slabsconfig:
            list of dictionaries. Each dictionary as a database key `db` and
            as many conditions to filter the database

        '''

        config = self.config
        slit = self.slit
        verbose = self.verbose

        slabs = {}  # slabs
        fconds = {}   # closest conditions found in database

        for slabname, slabcfg in slabsconfig.items():
            cfg = slabcfg.copy()
            
            
            if self.mode == 'database':
            
                dbi = cfg.pop('db')
        #        cfg['verbose'] = verbose
    
                si = dbi.get_closest(scale_if_possible=True, verbose=verbose, **cfg)
        #        try:
        #            del slabcfg['verbose']
        #        except KeyError:
        #            pass
    
            elif self.mode == 'calculate':
                
                sfi = cfg.pop('factory')
                si = sfi.eq_spectrum(**cfg)
                
            else:
                raise ValueError('Unknown mode: {0}'.format(self.mode))

            fcondsi = {}
            for k in cfg:
                fcondsi[k] = si.conditions[k]
            slabs[slabname] = si.copy()
            fconds[slabname] = fcondsi

        s = config(**slabs)
        s.apply_slit(slit, **self.slit_options)

        return s, slabs, fconds


