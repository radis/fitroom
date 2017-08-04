# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:27:38 2017

@author: erwan
"""

from neq.misc import norm_on

def Normalizer(wmin, wmax, how='mean'):
    return lambda w, a: norm_on(a, w, wmin=wmin, wmax=wmax, how=how)
