Installation
=============


It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed, as FairPy is updated frequently:

.. code-block:: bash

   pip install fairpy            # normal install
   pip install --upgrade fairpy  # or update if needed


Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/fairpy-team/fairpy.git
   cd fairpy
   pip install .

**Required Dependencies**\ :

* python>=3.8.0
* numpy>=1.21.5
* pandas>=1.4.3
* scikit-learn>=1.2.0
* gdown>=4.5.1
* xlrd>=2.0.1
* cvxopt>=1.3.0
* cvxpy>=1.3.0
* networkx>=2.8.7
* torch>=1.12.0