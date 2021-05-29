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
from radis.tools.database import SpecList, _scalable_inputs
from radis.spectrum.compare import get_residual
from radis import SpectrumFactory
from radis.misc.debug import printdbg
from radis.misc.basics import is_float
import sys
import pandas as pd


class SlabsConfigSolver():
    '''
    Machinery related to solving a specific Slabs configuration: parse the database,
    get the correct slab input, then calls the appropriate functions in neq.spec engine
    '''

    def __init__(self, config, source=None,
                 s_exp=None,
                 plotquantity='radiance', unit='mW/cm2/sr/nm',
                 slit=None, slit_options='default',
                 crop=None, retrieve_mode='safe', 
                 verbose=True, retrieve_error='ignore'):
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
        
        get_closest: bool
            when retrieving spectra from database, get cloest if set to True. Else
            get unique. 

        slit_options:
            if ``'default'``, use::

                {'norm_by':'area', 'shape':'triangular', 'unit':'nm', 'verbose':False}

            and adapt ``'shape'`` to ``'trapezoidal'`` if a tuple was given for slit

        crop: tuple, or None
            if not ``None``, restrain to the given fitted interval.

        retrieve_error: 'ignore', 'raise'
            if Spectrum cannot be calculated or retrieved from Database, then
            returns ``None`` as a Spectrum object. The rest of the code should
            deal with it. Else, raises an error immediatly. 
        
        retrieve_mode: 'safe', 'strict', 'closest'
            how to retrieve spectra when reading from database:
                
                - if 'strict', only retrieve the spectra that exactly match 
                the given conditions (allow scaling path_length or mole_fraction, still)
                
                - if 'safe', requires an exact match for all conditions (as in 
                'strict'), except for the 2 user defined variable conditions 
                ``xparam`` and ``yparam`` 
                
                - if 'closest', retrieves the closest spectrum in the database 
                
                    .. warning:: 
                        'closest' can induce user errors!!!. Ex: a Trot=1500 K 
                        spectrum can be used instead of a Trot=1550 K spectrum
                        if the latter is not available, without user necessarily
                        noticing. If you have any doubt, print the conditions
                        of the spectra used in the tools. Ex::
                            
                            for s in gridTool.spectra:
                                print(s)
                

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
        assert retrieve_error in ['ignore', 'raise']
        self.retrieve_error = retrieve_error    #: str: 'ignore', 'raise'

        # Get slit defaults
        self.slit = slit
        if slit_options == 'default':
            if isinstance(slit, tuple):
                slit_options = {'norm_by': 'area',
                                'shape': 'trapezoidal', 'unit': 'nm'}
            else:
                slit_options = {'norm_by': 'area',
                                'shape': 'triangular', 'unit': 'nm'}
        if 'verbose' not in slit_options:
            slit_options['verbose'] = False
        self.slit_options = slit_options

        self.crop = crop
        if crop is not None:
            raise NotImplementedError('crop not defined yet. Better crop the experimental spectrum directly')
        
        # Database retrieve options
        self.retrieve_mode = retrieve_mode   #: str 'safe', 'strict', 'closest'
        
        self.verbose = verbose

        # not a public option, but can be changed manually
        self.save_rescaled_bands = False

        self.source = source

        self.fitroom = None

    def connect(self, fitroom):
        ''' Triggered on connection to FitRoom '''
        self.fitroom = fitroom         # type: FitRoom

    def get_residual(self, s, normalize=False, normalize_how='max'):
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

        s: Spectrum object
            simulated spectrum to compare with (stored) experimental spectrum

        normalize: bool
            not implemented yet   # TODO

        Notes
        -----

        Implementation:

        interpolate experimental is harder (because of noise, and overlapping)
        we interpolate each new spectrum on the experiment

        '''

        plotquantity = self.plotquantity
        
        return get_residual(self.s_exp, s, plotquantity, ignore_nan=True, 
                            normalize=normalize, normalize_how=normalize_how)

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

            if source == 'database':

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'factory' in cfg:
                    warn('`factory` key dismissed in `database` source mode')
                if 'bandlist' in cfg:
                    warn('`database` source mode used but `bandlist` is given')

                dbi = cfg.pop('db')    # type: SpecDatabase
                
#                # Split columns in database if needed
#                split_columns_list = []
#                for k, v in cfg.items():
#                    if isinstance(v, tuple):
#                        split_columns_list.append(k)
#                if split_columns_list:
#                    dbi.df = expand_columns(dbi.df, split_columns_list)
                
                try:
                    if self.retrieve_mode == 'strict':
                        si = dbi.get_unique(scale_if_possible=True, verbose=verbose,
                                             **cfg)
                    elif self.retrieve_mode == 'safe':
                        # all parameters are enforced, except the 2 chosen by the user (xparam, yparam)
                        if slabname == self.fitroom.slbInteractx:    
                            cfg_fixed = {k:v for k,v in cfg.items() if k not in 
                                         _scalable_inputs+[self.fitroom.xparam]}
                        elif slabname == self.fitroom.slbInteracty:    
                            cfg_fixed = {k:v for k,v in cfg.items() if k not in 
                                         _scalable_inputs+[self.fitroom.yparam]}
                        else:    
                            cfg_fixed = {k:v for k,v in cfg.items() if k not in 
                                         _scalable_inputs}
                        # Get spectra corresponding to fixed parameters 
                        slist = dbi.get(verbose=False, **cfg_fixed)
                        if len(slist) == 0:
                            # give more insights:
                            dbi.get_closest(scale_if_possible=True, verbose=True, **cfg_fixed)
                            raise ValueError('Spectrum not found with these conditions. '+\
                                             'See closest spectrum above')
                        # Within this list, get the closest ones
                        si = SpecList(*slist).get_closest(scale_if_possible=True, verbose=verbose, 
                                             **cfg)
                    elif self.retrieve_mode == 'closest':
                        si = dbi.get_closest(scale_if_possible=True, verbose=verbose, 
                                             **cfg)
                    else:
                        raise NotImplementedError(self.retrieve_mode)
                except:
                    print(('An error occured while retrieving Spectrum from database: \n{0}'.format(
                            sys.exc_info())))
                    if self.retrieve_error == 'ignore':
                        si = None
                    else: # self.retrieve_error == 'raise':
                        raise

            elif source == 'calculate':
                
                # hack. get it working with multi Tvib.
                if 'Tvib1' in cfg and 'Tvib2' in cfg and 'Tvib3' in cfg and 'Tvib' not in cfg:
                    Tvib1 = cfg.pop('Tvib1')
                    Tvib2 = cfg.pop('Tvib2')
                    Tvib3 = cfg.pop('Tvib3')
                    cfg['Tvib'] = (Tvib1, Tvib2, Tvib3)

                if 'overpopulation' in cfg:
                    warn('`overpopulation` not used if not in from_bands source mode')
                if 'database' in cfg:
                    warn('`database` key dismissed in `calculate` source mode')
                if 'bandlist' in cfg:
                    warn('`calculate` source mode used but `bandlist` is given')

                sfi = cfg.pop('factory')        # type: SpectrumFactory
                if 'Tvib' in cfg and 'Trot' in cfg:
                    si = sfi.non_eq_spectrum(**cfg)
                elif 'Tgas' in cfg:
                    si = sfi.eq_spectrum(**cfg)
                else:
                    raise ValueError('Please give temperatures. Got: {0}'.format(cfg))

            elif source == 'calculate_noneq':
                raise DeprecationWarning("Use source='calculate' now")
#
#                if 'overpopulation' in cfg:
#                    warn('`overpopulation` not used if not in from_bands source mode')
#                if 'database' in cfg:
#                    warn('`database` key dismissed in `calculate_noneq` source mode')
#                if 'bandlist' in cfg:
#                    warn('`calculate_noneq` source mode used but `bandlist` is given')
#
#                sfi = cfg.pop('factory')        # type: SpectrumFactory
#                si = sfi.non_eq_spectrum(**cfg)

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
        # Keep initial conditions in output Spectrum
        if len(slabs) > 1 :
            for slabname, si in slabs.items():
                s.conditions['{0}'.format(slabname)] = '*'*80
                for k, v in si.conditions.items():
                    s.conditions['{0}_{1}'.format(slabname, k)] = v
        
        # (for developers: helps IDE find autocompletion)
#        assert isinstance(s, Spectrum)

        if slit:
            s.apply_slit(slit, **self.slit_options)

        return s, slabs, fconds


#def expand_columns(df, split_columns, verbose=True):
#    ''' Split columns in database if needed 
#    
#    Parameters
#    ----------
#    
#    df: pandas DataFrame
#        conditions of spectrum database 
#    '''
#    
#    assert isinstance(split_columns, list)
#
#    # Deal with multiple columns
#    for splitcolumn in split_columns:
#        if splitcolumn in df.columns:
#            dsplit = df[splitcolumn].str.split(',', expand=True)
#            dsplit.rename(columns={k:splitcolumn+str(k+1) for k in dsplit.columns}, 
#                                  inplace=True)
#            dsplit = dsplit.apply(pd.to_numeric)
##            df = pd.concat([df, dsplit], ignore_index=True).reindex(df.index)  # hopefully keep first index?
#            if not any([c in df.columns for c in dsplit.columns]):
#                df = pd.concat([df, dsplit], axis=1)
#            else:
#                if verbose:
#                    print('Columns already expanded: {0}'.format([c for c in dsplit.columns
#                          if c in df.columns]))
#            
#        else:
#            raise KeyError('{0} given in split_columns but not in Database: {1}'.format(
#                    splitcolumn, df.columns))
##                    if splitcolumn in kwconditions:
##                        v = kwconditions.pop(splitcolumn)
##                        try:
##                            v = v.split(',')
##                        except AttributeError:
##                            raise AttributeError('Key {0} is expected to be '.format(splitcolumn)+\
##                                                 'a comma separated string. Got {0}'.format(v))
##                        for k, vi in enumerate(v):
##                            kwconditions[splitcolumn+str(k+1)] = vi
##                    else:
##                        raise KeyError('{0} given in split_columns but not in conditions: {1}'.format(
##                                splitcolumn, dg.columns))
#            
#    return df
#
#                
