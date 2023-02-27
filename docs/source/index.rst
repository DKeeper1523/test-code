.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: fairpy_logo.png
    :scale: 30%
    :alt: logo

----


FairPy is a comprehensive **Python Library** for **Machine Learning Fairness**, covering various fairness notions, data structures, and learning tasks.
TODO: This challenging field has many key Applications....

FairPy includes more than **10** latest fairness algorithms, such as InflFair (ICML'22) and FairGLM (ICML'21).
For consistency and accessibility, FairPy is developed on top of `scikit-learn <https://scikit-learn.org/>`_ and `PyTorch <https://pytorch.org/>`_.


**Why FairPy: Compare to other libraries**:

| Attribute               | FairPy  | Fairlearn | AI Fairness 360 | inFairness |
|:------------------------|:-------:|:---------:|:---------------:|:----------:|
| Group Fairness          | &check; |  &check;  |     &check;     |  &cross;   |
| Individual Fairness     | &check; |  &cross;  |     &check;     |  &check;   |
| Minimax Fairness        | &check; |  &cross;  |     &cross;     |  &cross;   |
| Tabular Data            | &check; |  &check;  |     &check;     |  &check;   |
| Graph Data              | &check; |  &cross;  |     &cross;     |  &cross;   |
| Classification          | &check; |  &check;  |     &check;     |  &check;   |
| Regression              | &check; |  &cross;  |     &check;     |  &cross;   |
| Ranking                 | &check; |  &cross;  |     &cross;     |  &cross;   |
| Number of Algorithms    |   15+   |     4     |       14        |     3      |
| Compatible with sklearn | &check; |  &check;  |    Partially    |  &cross;   |
| Latest release          |    ?    | July 2021 |    March 2021   |  June 2022 |


**Outlier Detection Using PyGOD with 5 Lines of Code**\ :


.. code-block:: python

    from fairpy.dataset import Adult
    from fairpy.model import LabelBias

    model = LabelBias()     # Choose a fairness algorithm
    dataset = Adult()       # Choose or customize a dataset
    X, y, s = dataset.get_X_y_s()   # Extract features, labels, and sensitive attributes from the dataset

    model.fit(X, y, s)      # Fit model with provided data
    pred = model.predict(X) # Predict



Implemented Algorithms
======================

PyGOD toolkit consists of two major functional groups:

**(i) Node-level detection** :

| Type | Name in FairPy | Task           | Year | Ref. |
|:----:|:--------------:|:--------------:|:----:|:----:|
| Pre  | reweigh        | Any            | 2012 | [1]  |
| Pre  | LabelBias      | Classification | 2019 | [2]  |
| Pre  | LinearFairERM  | Any            | 2018 | [3]  |
| Pre  | DIRemover      | Any            | 2015 | [4]  |
| Pre  | IFair          | Any            | 2019 | [10] |
| Pre  | InflFair       | Classification | 2022 | [9]  |
| In   | FairCstr       | Classification | 2017 | [5]  |
| In   | FairGLM        | Classification | 2022 | [7]  |
| In   | FairPGRank     | Ranking        | 2019 | [12] |
| Post | EqOddsCalib    | Classification | 2016 | [6]  |
| Post | FairRank       | Ranking        | 2018 | [11] |


**(ii) Utility functions** :

TODO

----



**Reference**:

[1] Data preprocessing techniques for classification without discrimination  
[2] Identifying and Correcting Label Bias in Machine Learning  
[3] Empirical risk minimization under fairness constraints  
[4] Certifying and Removing Disparate Impact  
[5] Fairness Constraints: Mechanisms for Fair Classification  
[6] Equality of opportunity in supervised learning  
[7] Fair Generalized Linear Models with a Convex Penalty  
[8] The Fairness of Risk Scores Beyond Classification: Bipartite Ranking and the xAUC Metric  
[9] Achieving Fairness at No Utility Cost via Data Reweighing with Influence  
[10] iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making  
[11] FA*IR: A Fair Top-k Ranking Algorithm  
[12] Policy Learning for Fairness in Ranking  

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API References

   api_cc
   fairpy

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   team