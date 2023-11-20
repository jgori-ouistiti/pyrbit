.. ef:

Exponential Forgetting 
=======================


The Exponential Forgetting Model
---------------------------------

The exponential forgetting (EF) model follows from Ebbinghaus' forgetting curve.
According to this model, the probability :math:`p` for a subject to correctly recall an item is given by

.. math::

    p = \exp{(-\alpha (1-\beta)^k \Delta t)}, \quad \alpha \in \mathbb{R}^+, \beta \in [0,1], 

where :math:`\alpha` and :math:`\beta` are two memory parameters, :math:`k` is the number of times the item has been previously seen by the subject (repetition number) and :math:`\Delta_t` is the time that has elapsed since the last time that item was seen by the subject (delay). The parameter :math:`\alpha` (positive) is the initial forgetting rate: the lower :math:`\alpha`, the slower the forgetting and the better the recall. This rate gets reduced by :math:`(1-\beta)` (between 0 and 1) with each repetition which further slows the rate of forgetting, and thus the higher the :math:`\beta`, the better the recall.


The EF class
---------------

See the API Reference


Worked out Example with the EF model
--------------------------------------

.. literalinclude:: ../pyrbit/ef.py
    :start-after: [startdoc]
    :dedent: 4
