# FairPy: A Python Library for Machine Learning Fairness

FairPy is a comprehensive Python Library for Machine Learning Fairness, covering various fairness notions, data
structures, and learning tasks.

```python
# Select a fair algorithm and train
from fairpy import model

clf = model(*args, **kwargs)
clf.fit(X_train)
clf.predict(X_test)
```

## Team

1. Peizhao Li: Algorithms, Datasets, and Evaluations (Project Initiation - Now)
1. Han Yue: Test and Library Development (Project Initiation - Now)
1. Zexia Lu: Technical Product Manager (08.22 - Now)
1. Hongfu Liu: Advisor (Project Initiation - Now)

## Hiring Plan

1. ~~Technical Product Manager~~
1. Doc and Web Developer

## Development Plan

[Weekly Meeting Google Doc](https://docs.google.com/document/d/1Jb7dnB8Tu1-Uf-FQ7ZKZyrns984v1agQ9ktjdODhICU/edit?usp=sharing)

1. Preliminary: 1 - 1.5 months
    - [x] Implement one dataset, one classification algorithm, and one metric.
    - [x] Investigate other open-source projects
    - [x] Unit test
    - [x] Setup GitHub workflow
    - [ ] Project packaging experiments
    - [x] Hire a technical product manager

1. Scale: 3 months
    - [x] 6-7 datasets
    - [ ] 15+ algorithms
    - [ ] Various evaluation metrics
    - [ ] Comprehensive unit tests
    - [ ] Documentation

1. Alpha release
    - [ ] Collect feedbacks
   
1. Beta release

---

## Why FairPy: Compare to other libraries

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

AIF360:  
Classification Algorithms = 10; Regression Algorithms = 1; Ranking Alg. = 0;  
Algorithms support multiple sensitive attributes = All except PrejudiceRemover, MetaFairClassifier, DisparateImpactRemover;  <br>
One sensitive attributes with multiple values = All.

FairLearn:
Classification Algorithms = 4; Regression ALgorithms = 3;  Ranking Alg. = 0;  <br>
Algorithms support multiple sensitive attributes = All;
One sensitive attributes with multiple values = All.

---

## Datasets

1. Adult Data Set
1. COMPAS Recidivism Risk Score Data and Analysis
1. Bank Marketing Data Set
1. Statlog (German Credit Data) Data Set
1. default of credit card clients Data Set
1. The Dutch Virtual Census of 2001 - IPUMS Subset
1. Open University Learning Analytics dataset
1. Xing Dataset

## Models

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

### Plan

1. iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making
1. Fairness of exposure in rankings
1. Delayed Impact of Fair Machine Learning
1. FairCanary: Rapid COntinuous Explainable Fairness

## Evaluations

1. Demographic Parity
1. Equal Opportunity
1. xAUC: Cross-Area Under the Curve

## Reference

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
