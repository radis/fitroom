
.. |logo_png| image:: fitroom_ico.png

*******
Fitroom
*******

An interactive interface for multi-dimensional fitting of emission & absorption spectra.


.. minigallery:: fitroom.FitRoom

Features :

- 2D fitting with :class:`~fitroom.grid3x3_tool.Grid3x3`
- See the contribution of each slab in the line-of-sight with :class:`~fitroom.multislab_tool.MultiSlabPlot`
- Update an instrumental slit function interactively with :class:`~fitroom.slit_tool.SlitTool`
- Dynamic fitted parameters, linked to one-another with arbitrary functions with :class:`~fitroom.room.DynVar`. E.g:: 

    Tvib = 1.5 * Trot
    # Or :
    Trot_Plasma_CO2 = Trot_Plasma_CO

- Spectra computed on the fly, or tabulated, powered by :py:mod:`radis`.
- Automatic fitting with :class:`~fitroom.slit_tool.SlitTool`

.. note:: 

    Fitroom is better used as a tool to explore the effect of different fitting parameters, before 
    designing your own automatic fitting routines.
    


.. video:: https://user-images.githubusercontent.com/16088743/120166810-4ac14980-c1fd-11eb-9dd5-8fb037db8793.mp4
   :autoplay:
   :nocontrols:
   :width: 800
   :height: 400



---------------------------------------------------------------------

* :ref:`modindex`

---------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/index
   api

