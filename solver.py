# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:32:23 2017

@author: erwan

Machinery related to solving a specific Slabs configuration: parse the database,
get the correct slab input, then calls the appropriate functions in neq.spec engine

"""

import numpy as np
from scipy.interpolate import splev, splrep
from warnings import warn

class SlabsConfigSolver():
    '''
    Machinery related to solving a specific Slabs configuration: parse the database,
    get the correct slab input, then calls the appropriate functions in neq.spec engine
    '''

    def __init__(self, config, source=None,
                 wexp=None, Iexpcalib=None, wexp_shift=0,
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 slit=None, slit_options={'norm_by':'area', 'shape':'triangular',
                                          'slit_unit':'nm'},
                 verbose=True):
        '''
        Input
        ------

        source: 'database', 'calculate', 'from_bands'
            Whether to calculate spectra from scratch, retrieve them from a database,
            or combine vibrational bands
            Mode can be overriden by a 'source' parameter in every slab

        '''

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

        self.save_rescaled_bands = False  # not a public option, but can be changed manually

        self.source = source

        self.fitroom = None
        
    def connect(self):
        ''' Triggered on connection to FitRoom '''
        pass

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

        w, I = s.get(plotquantity, wunit='nm', Iunit=unit)

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

            if 'source' in cfg:
                source = cfg.pop('source')
            else:
                source = self.source

            if source == 'database':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'factory' in cfg:
                    warn('`factory` key dismissed in `database` source mode')
                if 'bandlist' in cfg:
                    warn('`database` source mode used but `bandlist` is given')

                dbi = cfg.pop('db')

                si = dbi.get_closest(scale_if_possible=True, verbose=verbose, **cfg)

            elif source == 'calculate':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'database' in cfg:
                    warn('`database` key dismissed in `calculate` source mode')
                if 'bandlist' in cfg:
                    warn('`calculate` source mode used but `bandlist` is given')

                sfi = cfg.pop('factory')
                si = sfi.eq_spectrum(**cfg)

            elif source == 'calculate_noneq':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'database' in cfg:
                    warn('`database` key dismissed in `calculate_noneq` source mode')
                if 'bandlist' in cfg:
                    warn('`calculate_noneq` source mode used but `bandlist` is given')

                sfi = cfg.pop('factory')
                si = sfi.non_eq_spectrum(**cfg)

            elif source == 'from_bands':

                if not 'overpopulation' in cfg:
                    cfg['overpopulation'] = None
                    warn('`from_bands` source mode used but `overpopulation` not given')
                if 'database' in cfg:
                    warn('`database` key dismissed in `from_bands` source mode')
                if 'db' in cfg:
                    del cfg['db']

                sfi = cfg.pop('bandlist')
                cfg['save_rescaled_bands'] = self.save_rescaled_bands
                si = sfi.non_eq_spectrum(**cfg)
                del cfg['save_rescaled_bands']

            else:
                raise ValueError('Unknown source mode: {0}'.format(self.source)+\
                                 ' Use calculate, calculate_non_eq, database or '+\
                                 'from_bands')
                
            # Overwrite name
            si.name = slabname

            fcondsi = {}
            for k in cfg:
                fcondsi[k] = si.conditions[k]
            slabs[slabname] = si.copy()
            fconds[slabname] = fcondsi

        s = config(**slabs)
        s.apply_slit(slit, **self.slit_options)

        return s, slabs, fconds


