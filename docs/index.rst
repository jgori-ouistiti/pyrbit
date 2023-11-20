
Welcome to PyRBIT's documentation!
==============================================

Project description
=====================

This library facilitates model based evaluation and design of experiments that feature Recall-Based Interaction Techniques (RBIT).

Installing PyRBIT
=======================

The library can be installed directly with pip:

.. code-block:: bash
    
    python3 -m pip install pyrbit


Alternatively, you can also downlaod the source and build it using `poetry <https://python-poetry.org/>`_

.. code-block:: bash

    poetry install

What can you do with pyrbit?
================================

pyrbit is a set of utilities for people who want help with designing experiments for evaluating and comparing recall-based interaction techniques. Here is how this tool can help you:

* Use cognitive models for simulating human memory:
  + Exponential Forgetting Model
  + ACT-R model
* Utilies to simulate experiments: populations, experiments, trials 
* Evaluating the informativeness of a particular schedule
* Run power analyses for a particular schedule to test the difference between block recall percentages
  

   
.. toctree::
    :maxdepth: 1
    :caption: See also

	Home page <self>
    Exponential Forgetting Model <ef>
    ACT-R Model <actr>
    Memory Utilities <memory>
    Evaluating the informativeness of a schedule <information>
    Comparing block recall percentages, power analysis <design>
    API reference <_autosummary/pyrbit>



Known Caveats
=====================


Indices
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
