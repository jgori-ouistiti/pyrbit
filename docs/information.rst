.. information:

Evaluating the informativeness of a schedule
===============================================
The information submodule has utilities to evaluate the informativeness of a schedule, as given by the average cumulative Fisher information. For example, the code below can be used to evaluate the informativeness of an iid schedule.

.. literalinclude:: ../pyrbit/information.py
    :start-after: [information-start]
    :dedent: 4


Currently, this evaluation assumes the exponential forgetting model, but the ACT-R model could be included --- just file an issue!


