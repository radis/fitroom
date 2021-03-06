
.. |logo_png| image:: fitroom_ico.png

*******
Fitroom
*******

An interactive interface for multi-dimensional fitting of emission & absorption spectra.



.. video:: https://user-images.githubusercontent.com/16088743/120166810-4ac14980-c1fd-11eb-9dd5-8fb037db8793.mp4
   :autoplay:
   :nocontrols:
   :width: 800
   :height: 400



Features :
 
- 2D fitting with :class:`~fitroom.grid3x3_tool.Grid3x3`
- See the contribution of each slab in the line-of-sight with :class:`~fitroom.multislab_tool.MultiSlabPlot`
- Update an instrumental slit function interactively with :class:`~fitroom.slit_tool.SlitTool`
- Dynamic fitted parameters, linked to one-another with arbitrary functions with :class:`~fitroom.room.DynVar`. E.g:: 

    Tvib = 1.5 * Trot
    # Or :
    Trot_Plasma_CO2 = Trot_Plasma_CO

- Spectra computed on the fly, or tabulated, powered by :py:mod:`radis`.
- arbitrary line-of-sight and number of fitted parameters by fitting a combination of :py:func:`~radis.los.slabs.SerialSlabs` and :py:func:`~radis.los.slabs.MergeSlabs`
- Brute-force fitting with :class:`~fitroom.selection_tool.CaseSelector.precompute_residual`

.. note:: 

    Fitroom is better used as a tool to explore the effect of different fitting parameters, before 
    designing your own automatic fitting routines.
    

.. minigallery:: fitroom.FitRoom


See also the automatic fitting functions directly built in RADIS :

- :py:class:`~radis.lbl.factory.SpectrumFactory` 's :py:meth:`~radis.lbl.factory.SpectrumFactory.fit_spectrum`

- :py:class:`~radis.tools.database.SpecDatabase` 's :py:meth:`~radis.tools.database.SpecDatabase.fit_spectrum`



---------------------------------------------------------------------

* :ref:`modindex`

---------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/index
   api

