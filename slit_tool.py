# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:38:25 2017

@author: erwan
"""


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from neq.spec import plot_slit
from neq.misc import is_float
from six import string_types


class SlitTool():
    ''' Tool to manipulate slit function '''
    
    
    def __init__(self, slit_function):

        self.slit_function = slit_function
        
        self.fitroom = None
    
        fig, ax = plot_slit(slit_function)
        
        self.fig = fig
        self.ax = ax 
        
        if not isinstance(slit_function, string_types):
            # Add sliders
            if is_float(slit_function):
                base = slit_function
                top = 0
            elif isinstance(slit_function, tuple):
                base, top = max(slit_function), min(slit_function)
            else:
                raise ValueError('Wrong slit function format: {0}'.format(slit_function))
                        
            plt.axis([0, 1, -10, 10])
            
            axcolor = 'lightgoldenrodyellow'
            axtop = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            axbase = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

            sltop = Slider(axtop, 'Top', 0, 3*base, valinit=top)
            slbase = Slider(axbase, 'Base', 0, 3*base, valinit=base)
    
            sltop.on_changed(self.update)
            slbase.on_changed(self.update)
            
            self.sltop = sltop
            self.slbase = slbase
            
    def update(self):
        ''' update slit function '''
        
        top = self.sltop.val
        base = self.slbase.val
#        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#        fig.canvas.draw_idle()
        
        print('New slit assigned:', top, base)
        
        if self.fitroom is None:
            raise ValueError('Fitroom not connected')
            
        self.fitroom.solver.slit = (top, base)
