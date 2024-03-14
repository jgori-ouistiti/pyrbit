.. memory:

Memory Utilities
====================================
PyRBIT defines several utilities that deal with memory models and experiments, of which we present here 


The MemoryModel class
--------------------------
All memory models in PyRBIT inherit from the MemoryModel class, and have to redefine the ``update`` method (how is the state of the memory model update when ``item`` is shown at ``time``) and the ``compute_probabilities`` method (computes the probability of recall for all items at ``time``. If ``time`` is ``None``, should use the last time where any ``item`` was presented.)

.. literalinclude:: ../pyrbit/mem_utils.py
    :pyobject: MemoryModel


Schedules
------------
.. literalinclude:: ../pyrbit/mem_utils.py
    :start-after: [schedule]
    :end-before: [trial]
    :dedent: 4

Trials
---------
.. literalinclude:: ../pyrbit/mem_utils.py
    :start-after: [trial]
    :end-before: [population]
    :dedent: 4

Populations
-------------
.. literalinclude:: ../pyrbit/mem_utils.py
    :start-after: [population]
    :end-before: [experiment]
    :dedent: 4

Experiments
-------------
.. literalinclude:: ../pyrbit/mem_utils.py
    :start-after: [experiment]
    :dedent: 4
