
.. |logo_png| image:: radis_ico.png

*******
Fitroom
*******

An interactive Python interface for multi-dimensional fitting of emission & absorption spectra.

Features :

- 2D fitting 
- See the contribution of each slab in the line-of-sight 
- Update an instrumental slit function interactively
- Dynamic fitted parameters, linked to one-another with arbitrary functions. E.g:: 

    Tvib = 1.5 * Trot
    # Or :
    Trot_Plasma_CO2 = Trot_Plasma_CO

- Spectra computed on the fly, or tabulated, powered by :py:mod:`radis`.
- Automatic fitting 

.. note:: 

    Fitroom is better used as a tool to explore the effect of different fitting parameters, before 
    designing your own automatic fitting routines.
    


.. video:: https://user-images.githubusercontent.com/16088743/120166810-4ac14980-c1fd-11eb-9dd5-8fb037db8793.mp4
   :autoplay:
   :nocontrols:
   :width: 800
   :height: 400




.. minigallery:: fitroom



---------------------------------------------------------------------

* :ref:`modindex`

---------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/index
   api
