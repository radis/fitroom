# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:32:23 2017

@author: erwan

Machinery related to solving a specific Slabs configuration: parse the database,
get the correct slab input, then calls the appropriate functions in neq.spec engine

-------------------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy.interpolate import splev, splrep
from warnings import warn
from radis import SpecDatabase, Spectrum  # imported for static debugger
from radis.spectrum.compare import get_residual
from neq.spec import SpectrumFactory
from neq.misc.debug import printdbg
from neq.misc.basics import is_float
import sys


class SlabsConfigSolver():
    '''
    Machinery related to solving a specific Slabs configuration: parse the database,
    get the correct slab input, then calls the appropriate functions in neq.spec engine
    '''

    def __init__(self, config, source=None,
                 s_exp=None,
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 slit=None, slit_options='default',
                 verbose=True):
        '''
        Parameters
        ----------

        config: dict
            list of Slabs that represent the Spatial model (to solve RTE)

        source: 'database', 'calculate', 'from_bands'
            Whether to calculate spectra from scratch, retrieve them from a database,
            or combine vibrational bands
            Mode can be overriden by a 'source' parameter in every slab

        s_exp: :class:`~radis.spectrum.spectrum.Spectrum` 
            experimental spectrum

        plotquantity: 'radiance', 'transmittance_noslit', etc.


        Other Parameters
        ----------------

        slit_options:
            if ``'default'``, use::

                {'norm_by':'area', 'shape':'triangular',
                                                  'unit':'nm'}

            and adapt ``'shape'`` to ``'trapezoidal'`` if a tuple was given for slit


        Examples
        --------
        
        See the working case in :mod:`~neq.test.math.test_fitroom`. In particular, run
        :func:`~neq.test.math.test_fitroom.test_start_fitroom`
            
        '''

#        self.dbInteractx = dbInteractx
#        self.dbInteracty = dbInteracty

        self.config = config

        self.s_exp = s_exp
        if s_exp is not None:
            wexp, Iexpcalib = s_exp.get(plotquantity, Iunit=unit)
        self.wexp = wexp
        self.Iexpcalib = Iexpcalib
        self.plotquantity = plotquantity
        self.unit = unit

        # Get slit defaults
        self.slit = slit
        if slit_options == 'default':
            if isinstance(slit, tuple):
                slit_options = {'norm_by': 'area',
                                'shape': 'trapezoidal', 'unit': 'nm'}
            else:
                slit_options = {'norm_by': 'area',
                                'shape': 'triangular', 'unit': 'nm'}
        self.slit_options = slit_options

        self.verbose = verbose

        # not a public option, but can be changed manually
        self.save_rescaled_bands = False

        self.source = source

        self.fitroom = None

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def get_residual(self, s, norm='not_implemented'):
        ''' Returns difference between experimental and simulated spectra
        By default, uses :func:`~radis.spectrum.compare.get_residual` function
        You can change the residual by overriding this function. 

        Examples
        --------

        Replace get_residual with new_residual::

            solver.get_residual = lambda s: new_residual(solver.s_exp, s, 'radiance')

        Note that default solver would be written::

            from radis import get_residual
            solver.get_residual = lambda s: get_residual(solver.s_exp, s,
                                                         solver.plotquantity,
                                                         ignore_nan=True)

        Parameters
        ----------

        Parameters
        ----------

        s: Spectrum object
            simulated spectrum to compare with (stored) experimental spectrum

        norm not implemented yet
        # TODO

        Notes
        -----

        Implementation:

        interpolate experimental is harder (because of noise, and overlapping)
        we interpolate each new spectrum on the experiment

        '''

        plotquantity = self.plotquantity
        return get_residual(self.s_exp, s, plotquantity, ignore_nan=True)

#        wexp = self.wexp
#        Iexpcalib = self.Iexpcalib
#        plotquantity = self.plotquantity
#        unit = self.unit
#
#
#        b = np.argsort(wexp)
#        wsort, Isort = wexp[b], Iexpcalib[b]
#
#        w, I = s.get(plotquantity, wunit='nm', Iunit=unit)
#
#        # crop to overlapping range
#        b = (wsort>w.min()) & (wsort<w.max())
#        wsort, Isort = wsort[b], Isort[b]
#        if len(wsort) == 0:
#            # no overlap between calculated and exp spectra ?
#            if __debug__: printdbg('no overlap in get_residual() ? ')
#            return np.nan
#        b = (w>wsort.min()) & (w<wsort.max())
#        w, I= w[b], I[b]
#
#        if w[0]>w[-1]:
#            w, I = w[::-1], I[::-1]
#
#        tck = splrep(w, I)
#        Iint = splev(wsort, tck)
#
#    #    error = np.sqrt(np.trapz(np.abs((Iint-Isort)/(Iint+Isort)), x=wsort).sum())
#        error = np.sqrt(np.trapz(np.abs(Iint-Isort), x=wsort).sum())
#
#        return error

    def calc_slabs(self, **slabsconfig):
        '''
        Parameters
        ----------

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
                source = cfg.pop('source')       # type: str
            else:
                source = self.source
            if 'Tvib1' in cfg and 'Tvib2' in cfg and 'Tvib3' in cfg and 'Tvib' not in cfg:
                Tvib1 = cfg.pop('Tvib1')
                Tvib2 = cfg.pop('Tvib2')
                Tvib3 = cfg.pop('Tvib3')
                cfg['Tvib'] = (Tvib1, Tvib2, Tvib3)

            if source == 'database':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'factory' in cfg:
                    warn('`factory` key dismissed in `database` source mode')
                if 'bandlist' in cfg:
                    warn('`database` source mode used but `bandlist` is given')

                dbi = cfg.pop('db')    # type: SpecDatabase

                try:
                    si = dbi.get_closest(
                        scale_if_possible=True, verbose=verbose, **cfg)
                except:
                    print(('An error occured while retrieving Spectrum from database: \n{0}'.format(
                        sys.exc_info())))
                    si = None

            elif source == 'calculate':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'database' in cfg:
                    warn('`database` key dismissed in `calculate` source mode')
                if 'bandlist' in cfg:
                    warn('`calculate` source mode used but `bandlist` is given')

                sfi = cfg.pop('factory')        # type: SpectrumFactory
                si = sfi.eq_spectrum(**cfg)

            elif source == 'calculate_noneq':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'database' in cfg:
                    warn('`database` key dismissed in `calculate_noneq` source mode')
                if 'bandlist' in cfg:
                    warn('`calculate_noneq` source mode used but `bandlist` is given')

                sfi = cfg.pop('factory')        # type: SpectrumFactory
                si = sfi.non_eq_spectrum(**cfg)

            elif source == 'from_bands':

                if not 'overpopulation' in cfg:
                    cfg['overpopulation'] = None
                    warn('`from_bands` source mode used but `overpopulation` not given')
                if 'database' in cfg:
                    warn('`database` key dismissed in `from_bands` source mode')
                if 'db' in cfg:
                    del cfg['db']

                sfi = cfg.pop('bandlist')        # type: BandList
                cfg['save_rescaled_bands'] = self.save_rescaled_bands
                si = sfi.non_eq_spectrum(**cfg)
                del cfg['save_rescaled_bands']

            elif source == 'constants':
                # used for global variables.
                # Just update the config file
                #                slabs[slabname] = None
                fconds[slabname] = cfg
                continue

            else:
                raise ValueError('Unknown source mode: {0}'.format(self.source) +
                                 ' Use calculate, calculate_non_eq, database or ' +
                                 'from_bands')

            # Get spectrum calculation output
            if si is None:  # ex: user asked for negative path length
                warn('Spectrum couldnt be calculated')
                return (None, None, None)

            else:
                # Overwrite name
                si.name = slabname

                fcondsi = {}
                for k in cfg:
                    fcondsi[k] = si.conditions[k]
                # Placeholder
                if 'Tvib' in fcondsi:
                    Tvib = fcondsi['Tvib']
                    if not is_float(Tvib):
                        fcondsi['Tvib1'] = Tvib[0]
                        fcondsi['Tvib2'] = Tvib[1]
                        fcondsi['Tvib3'] = Tvib[2]

                slabs[slabname] = si.copy()
                fconds[slabname] = fcondsi
                
        # Add the value we want, recalculate if needed
        required_quantities = set(['transmittance_noslit', 'radiance_noslit']) 
        # TODO: deal with case where self.plotquantity is not any of the above, 
        # or a convoluted value of them
        for required_quantity in required_quantities:
            for _, si in slabs.items():
                si.update(required_quantity, verbose=False)
            
        # Calculate the Line of Sight model
        s = config(**slabs)
        # (for developers: helps IDE find autocompletion)
        assert isinstance(s, Spectrum)

        s.apply_slit(slit, verbose=False, **self.slit_options)

        return s, slabs, fconds
