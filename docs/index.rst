.. figure:: fairpy_logo.pdf
    :scale: 50%
    :alt: logo

----

FairPy Documentation
=====================

FairPy is a comprehensive **Python Library** for **Machine Learning Fairness**, covering various fairness notions, data structures, and learning tasks.
TODO: This challenging field has many key Applications....

FairPy includes more than **10** latest fairness algorithms, such as InflFair (ICML'22) :cite:p:`li2022achieving` and FairGLM (ICML'22) :cite:p:`do2022fair`.
For consistency and accessibility, FairPy is developed on top of `scikit-learn <https://scikit-learn.org/>`_ and `PyTorch <https://pytorch.org/>`_.

----

Why FairPy: Compare to Other Libraries
=======================================

=======================  ==========  =========  ===============  ==========
Attribute                FairPy      Fairlearn  AI Fairness 360  inFairness
=======================  ==========  =========  ===============  ==========
Group Fairness           [x]         [x]        [x]              [ ]       
Individual Fairness      [x]         [ ]        [x]              [x]    
Minimax Fairness         [x]         [ ]        [ ]              [ ]       
Tabular Data             [x]         [x]        [x]              [x]    
Graph Data               [x]         [ ]        [ ]              [ ]       
Classification           [x]         [x]        [x]              [x]    
Regression               [x]         [ ]        [x]              [ ]       
Ranking                  [x]         [ ]        [ ]              [ ]       
Number of Algorithms     10+         4          14               3
Compatible with sklearn  [x]         [x]        Partially        [ ]       
Latest release           March 2023  July 2021  March 2021       June 2022
=======================  ==========  =========  ===============  ==========

----

Sample Code of FairPy
======================


.. code-block:: python

    from fairpy.dataset import Adult
    from fairpy.model import LabelBias

    model = LabelBias()     # Choose a fairness algorithm
    dataset = Adult()       # Choose or customize a dataset
    X, y, s = dataset.get_X_y_s()   # Extract features, labels, and sensitive attributes from the dataset

    model.fit(X, y, s)      # Fit model with provided data
    pred = model.predict(X) # Predict

----


Implemented Algorithms
=======================

FairPy toolkit consists of three major functional groups:

**(i) Fairness Algorithms** :

====  ==============  ==============  ====  ====
Type  Name in FairPy  Task            Year  Ref
====  ==============  ==============  ====  ====
Pre   reweigh         Any             2012  :cite:p:`kamiran2012data`
Pre   LabelBias       Classification  2020  :cite:p:`jiang2020identifying`
Pre   LinearFairERM   Any             2018  :cite:p:`donini2018empirical`
Pre   DIRemover       Any             2015  :cite:p:`feldman2015certifying`
Pre   IFair           Any             2019  :cite:p:`lahoti2019ifair`
Pre   InflFair        Classification  2022  :cite:p:`li2022achieving`
In    FairCstr        Classification  2017  :cite:p:`zafar2017fairness`
In    FairGLM         Classification  2022  :cite:p:`do2022fair`
In    FairPGRank      Ranking         2019  :cite:p:`singh2019policy`
Post  EqOddsCalib     Classification  2016  :cite:p:`hardt2016equality`
Post  FairRank        Ranking         2017  :cite:p:`zehlike2017fa`
====  ==============  ==============  ====  ====

**(ii) Datasets** :

=====  ==============
Type   Name in FairPy
=====  ==============
Table  Adult
Table  Bank
Table  Compas
Table  Credit
Table  Dutch
Table  German
Table  Oulad
Table  Xing
=====  ==============


**(iii) Metrics** :

==============  ==============  
Type            Name in FairPy
==============  ==============
Classification  binary_dp
Classification  binary_eop
Ranking         xAUC
Ranking         dcg
==============  ==============


----

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   fairpy.model
   fairpy.dataset

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   reference