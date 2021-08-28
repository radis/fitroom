# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:27:38 2017

@author: erwan
"""

from __future__ import absolute_import
from radis.misc.arrays import norm_on


def Normalizer(wmin, wmax, how='mean'):
    """
    Examples
    --------
    
    .. minigallery:: fitroom.tools.Normalizer

    """
    return lambda w, a: norm_on(a, w, wmin=wmin, wmax=wmax, how=how)
