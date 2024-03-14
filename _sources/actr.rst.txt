.. actr:

ACT-R
=======================


The ACT-R model
---------------------------------

We use one of the ACT-R model described in [1]_. The model reads

.. math::

    m & = \log \left( \sum_{i=1}^n \Delta t_i^{-d} \right)                                \\
    p & = \left( 1 + \exp\left(\frac{\tau -m }{s}\right) \right)^{-1}


where the activation, :math:`m` , is a power function of the times :math:`\Delta t_i` since the item :math:`i` was presented, and the exponent :math:`d` stands for delay, and :math:`s` and :math:`\tau` are parameters that add extra flexibility to the model.
In comparison with the EF model where recall is just a function of when the last item was presented, the ACT-R model accounts for all past item presentations via :math:`m`. 

The ACTR class
----------------

See the API Reference


Worked out Example with the ACT-R model
-----------------------------------------

.. literalinclude:: ../pyrbit/actr.py
    :start-after: [startdoc]
    :dedent: 4




.. [1] Pavlik Jr, Philip I., and John R. Anderson. "Practice and forgetting effects on vocabulary memory: An activation‐based model of the spacing effect." Cognitive science 29.4 (2005): 559-586.
