.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: fairpy_logo.png
    :scale: 50%
    :alt: logo

----

FairPy Documentation
====================

FairPy is a comprehensive **Python Library** for **Machine Learning Fairness**, covering various fairness notions, data structures, and learning tasks.
TODO: This challenging field has many key Applications....

FairPy includes more than **10** latest fairness algorithms, such as InflFair (ICML'22) and FairGLM (ICML'22).
For consistency and accessibility, FairPy is developed on top of `scikit-learn <https://scikit-learn.org/>`_ and `PyTorch <https://pytorch.org/>`_.

----

Why FairPy: Compare to Other Libraries
====================

.. |check| raw:: html
    <input checked=""  type="checkbox">
    
.. |uncheck_| raw:: html
    <input disabled="" type="checkbox">

=======================  =======  =========  ===============  ==========
Attribute                FairPy   Fairlearn  AI Fairness 360  inFairness
=======================  =======  =========  ===============  ==========
Group Fairness           |check|  |check|    |check|          |uncheck_|
Individual Fairness      |check|  |uncheck_| |check|          |check|
Minimax Fairness         |check|  |uncheck_| |uncheck_|       |uncheck_|
Tabular Data             |check|  |check|    |check|          |check|
Graph Data               |check|  |uncheck_| |uncheck_|       |uncheck_|
Classification           |check|  |check|    |check|          |check|
Regression               |check|  |uncheck_| |check|          |uncheck_|
Ranking                  |check|  |uncheck_| |uncheck_|       |uncheck_|
Number of Algorithms     10+      4          14               3
Compatible with sklearn  |check|  |check|    Partially        |uncheck_|
Latest release           ?        July 2021  March 2021       June 2022
=======================  =======  =========  ===============  ==========

----

Sample Code of FairPy
====================


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
====================

FairPy toolkit consists of three major functional groups:

**(i) Fairness Algorithms** :

====  ==============  ==============  ====  ====
Type  Name in FairPy  Task            Year  Ref
====  ==============  ==============  ====  ====
Pre   reweigh         Any             2021  [1]
Pre   LabelBias       Classification  2021  [2]
Pre   LinearFairERM   Any             2021  [3]
Pre   DIRemover       Any             2021  [4]
Pre   IFair           Any             2021  [10]
Pre   InflFair        Classification  2021  [9]
In    FairCstr        Classification  2021  [5]
In    FairGLM         Classification  2021  [7]
In    FairPGRank      Ranking         2021  [12]
Post  EqOddsCalib     Classification  2021  [6]
Post  FairRank        Ranking         2021  [11]
====  ==============  ==============  ====  ====

**(iii) Datasets** :

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



**Reference**:

* [1] Data preprocessing techniques for classification without discrimination  
* [2] Identifying and Correcting Label Bias in Machine Learning  
* [3] Empirical risk minimization under fairness constraints  
* [4] Certifying and Removing Disparate Impact  
* [5] Fairness Constraints: Mechanisms for Fair Classification  
* [6] Equality of opportunity in supervised learning  
* [7] Fair Generalized Linear Models with a Convex Penalty  
* [8] The Fairness of Risk Scores Beyond Classification: Bipartite Ranking and the xAUC Metric  
* [9] Achieving Fairness at No Utility Cost via Data Reweighing with Influence  
* [10] iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making  
* [11] FA*IR: A Fair Top-k Ranking Algorithm  
* [12] Policy Learning for Fairness in Ranking  

----

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API References

   fairpy.model
   fairpy.dataset

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   team