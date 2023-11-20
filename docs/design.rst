.. design:

Informing the Design of Experiments
=====================================
The design submodule has utilities to run comparative experiments. For example, to compare two RBITs, you can instantiate two population models with different parameters and call ``run_comparative_experiment``

.. literalinclude:: ../pyrbit/design.py
    :start-after: [run-comparative-experiment-start]
    :end-before: [run-comparative-experiment-after]
    :dedent: 4

You may also compute recall block percentages directly, as the next worked out example shows:

.. literalinclude:: ../pyrbit/design.py
    :start-after: [diff-block-percentage]
    :end-before: [power analysis]
    :dedent: 4



Power Analysis
---------------------------------
These evaluations can be used to perform an empirical power analysis. The following code shows how you can compute type 1 and type 2 errors for various experimental conditions and ways of combining p values. 
You can also define your own way to compute p values. We suggest considering the type 2 error of the combination which has the type 1 error closest to the nominal value (5%) (in the following example, it would be the Bonferroni-like method)


.. literalinclude:: ../pyrbit/design.py
    :start-after: [power analysis]
    :dedent: 4




